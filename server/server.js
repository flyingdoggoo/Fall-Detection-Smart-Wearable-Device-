require('dotenv').config();
const express=require('express');
const http=require('http');
const socketIO=require('socket.io');
const cors=require('cors');
const os=require('os');
const fs=require('fs');
const path=require('path');
const dgram=require('dgram');

const app=express();
const server=http.createServer(app);
const io=socketIO(server,{cors:{origin:'*',methods:['GET','POST']},pingInterval:25000,pingTimeout:60000,transports:['websocket','polling'],allowUpgrades:false});
const coap=dgram.createSocket('udp4');

app.use(cors());
app.use(express.json({limit:'10mb'}));

const PORT=Number(process.env.PORT||3000);
const COAP_PORT=Number(process.env.COAP_PORT||5683);
const DATA_DIR=path.join(__dirname,'data','collected');
const NORMAL_DIR=path.join(DATA_DIR,'Normal');
const FALL_DIR=path.join(DATA_DIR,'Fall');
const FEATURES_HEADER='window_time,magnitude_avg,sma,max_accel,max_gyro,std_accel,jerk_peak,bpm,fsm_state,fall_detected\n';
const COAP_TYPE={CON:0,NON:1,ACK:2};
const COAP_CODE={POST:2,CREATED:65,CHANGED:68,BAD_REQUEST:128,NOT_FOUND:132,METHOD_NOT_ALLOWED:133,CONFLICT:137,INTERNAL_SERVER_ERROR:160};
const SENSOR_PACKET_VERSION=2;
const LEGACY_SENSOR_PACKET_VERSION=1;
const SENSOR_SAMPLE_BYTES=28;

let connectedClients=0;
let currentLabel='1';
let currentSessionId=null;
let currentSessionLabel=null;
let sessionStartTime=null;
let fallMarkers=[];

const ensureDir=(dir)=>{ if(!fs.existsSync(dir)) fs.mkdirSync(dir,{recursive:true}); };
[DATA_DIR,NORMAL_DIR,FALL_DIR].forEach(ensureDir);

function inferLabelFromSessionId(sessionId){
  if(typeof sessionId!=='string') return null;
  const match=sessionId.match(/^label([01])_/);
  return match?match[1]:null;
}

function getSessionDir(sessionId=currentSessionId,label=currentSessionLabel){
  const effectiveLabel=label??inferLabelFromSessionId(sessionId)??currentLabel;
  return path.join(effectiveLabel==='1'?FALL_DIR:NORMAL_DIR,sessionId);
}

function getLabelText(label){ return String(label)==='1'?'FALL':'NORMAL'; }

function getLocalIP(){
  if(process.env.SERVER_IP) return process.env.SERVER_IP;
  const interfaces=os.networkInterfaces();
  const preferredPrefix=process.env.PREFERRED_IP_PREFIX||'192.168.1';
  let preferredIP=null;
  let fallbackIP=null;
  for(const name of Object.keys(interfaces)){
    for(const iface of interfaces[name]){
      if(iface.family==='IPv4'&&!iface.internal){
        if(iface.address.startsWith(preferredPrefix)) preferredIP=iface.address;
        if(!fallbackIP) fallbackIP=iface.address;
      }
    }
  }
  return preferredIP||fallbackIP||'localhost';
}

function stopCurrentSession(reason='stopped'){
  if(!currentSessionId) return null;
  const stoppedSessionId=currentSessionId;
  console.log(`✓ Session ${stoppedSessionId} ${reason} - ${fallMarkers.length} fall markers kept in memory only`);
  currentSessionId=null;
  currentSessionLabel=null;
  sessionStartTime=null;
  fallMarkers=[];
  return stoppedSessionId;
}

function startNewSession(label=currentLabel){
  const effectiveLabel=String(label)==='0'?'0':'1';
  const timestamp=new Date().toISOString().replace(/[:.]/g,'-').slice(0,-5);
  currentSessionId=`label${effectiveLabel}_${timestamp}`;
  currentSessionLabel=effectiveLabel;
  sessionStartTime=new Date().toISOString();
  fallMarkers=[];
  const sessionDir=getSessionDir(currentSessionId,currentSessionLabel);
  ensureDir(sessionDir);
  fs.writeFileSync(path.join(sessionDir,'accel.csv'),'accel_time_list,accel_x_list,accel_y_list,accel_z_list\n');
  fs.writeFileSync(path.join(sessionDir,'gyro.csv'),'gyro_time_list,gyro_x_list,gyro_y_list,gyro_z_list\n');
  fs.writeFileSync(path.join(sessionDir,'features.csv'),FEATURES_HEADER);
  console.log(`\n✓ New session started: ${currentSessionId} [Label: ${effectiveLabel} - ${getLabelText(effectiveLabel)}]`);
}

function getBatchElapsedSeconds(batchData){
  const startMs=Number(batchData.window_start_ms);
  if(Number.isFinite(startMs)) return startMs/1000;
  if(!sessionStartTime) return 0;
  return (Date.now()-new Date(sessionStartTime).getTime())/1000;
}

function saveToCSV(batchData){
  if(!currentSessionId) return;
  const sessionDir=getSessionDir();
  const accelPath=path.join(sessionDir,'accel.csv');
  const gyroPath=path.join(sessionDir,'gyro.csv');
  let accelLines='';
  for(const sample of batchData.accel_data) accelLines+=`${Number(sample.t).toFixed(6)},${Number(sample.x).toFixed(6)},${Number(sample.y).toFixed(6)},${Number(sample.z).toFixed(6)}\n`;
  fs.appendFileSync(accelPath,accelLines);
  let gyroLines='';
  for(const sample of batchData.gyro_data) gyroLines+=`${Number(sample.t).toFixed(6)},${Number(sample.x).toFixed(6)},${Number(sample.y).toFixed(6)},${Number(sample.z).toFixed(6)}\n`;
  fs.appendFileSync(gyroPath,gyroLines);
}

function saveFeatures(batchData){
  if(!currentSessionId) return;
  const featuresPath=path.join(getSessionDir(),'features.csv');
  const f=batchData.features||{};
  const elapsed=getBatchElapsedSeconds(batchData);
  const line=[
    elapsed.toFixed(6),
    Number(f.magnitude_avg??0).toFixed(6),
    Number(f.sma??0).toFixed(6),
    Number(f.max_accel??0).toFixed(6),
    Number(f.max_gyro??0).toFixed(6),
    Number(f.std_accel??0).toFixed(6),
    Number(f.jerk_peak??0).toFixed(6),
    Number(batchData.bpm??0).toFixed(6),
    Number(batchData.fsm_state??0).toFixed(6),
    batchData.fall_detected?1:0
  ].join(',')+'\n';
  fs.appendFileSync(featuresPath,line);
}

function summarizeBatch(batchData){
  const magnitude=batchData.features?.magnitude_avg;
  const magText=Number.isFinite(magnitude)?magnitude.toFixed(2):'--';
  return `${batchData.window_size} samples | BPM: ${batchData.bpm??0} | Mag: ${magText}`;
}

function validateBatchData(batchData){
  if(!batchData||!Array.isArray(batchData.accel_data)||!Array.isArray(batchData.gyro_data)) return 'Missing accel_data or gyro_data';
  if(batchData.accel_data.length===0||batchData.gyro_data.length===0) return 'Empty sensor batch';
  if(batchData.accel_data.length!==batchData.gyro_data.length) return 'accel_data and gyro_data must have the same length';
  batchData.window_size=batchData.accel_data.length;
  const sampleIntervalMs=Number(batchData.sample_interval_ms);
  batchData.sample_interval_ms=Number.isFinite(sampleIntervalMs)&&sampleIntervalMs>0?sampleIntervalMs:20;
  const sampleRate=Number(batchData.sample_rate);
  batchData.sample_rate=Number.isFinite(sampleRate)&&sampleRate>0?sampleRate:Math.round(1000/batchData.sample_interval_ms);
  const windowStartMs=Number(batchData.window_start_ms);
  batchData.window_start_ms=Number.isFinite(windowStartMs)?windowStartMs:0;
  batchData.bpm=Number(batchData.bpm??0);
  batchData.ir_raw=Number(batchData.ir_raw??0);
  batchData.fsm_state=Number(batchData.fsm_state??0);
  batchData.fall_detected=Boolean(batchData.fall_detected);
  return null;
}

function emitFallDetected(batchData,source){
  if(!batchData.fall_detected) return;
  if(Number.isFinite(Number(batchData.chunk_index))&&Number(batchData.chunk_index)!==0) return;
  const elapsed=getBatchElapsedSeconds(batchData);
  const timestamp=new Date().toISOString();
  fallMarkers.push({timestamp,elapsed_seconds:elapsed,source});
  io.emit('fallDetected',{session_id:currentSessionId,timestamp,fsm_state:batchData.fsm_state,features:batchData.features});
  console.log(`🚨 Fall detected by ${source} at ${elapsed.toFixed(2)}s`);
}

function processBatchData(batchData,source){
  const validationError=validateBatchData(batchData);
  if(validationError) return {status:400,body:{success:false,error:validationError}};
  if(!currentSessionId) return {status:409,body:{success:false,error:'No active session - batch ignored'}};
  try{
    saveToCSV(batchData);
    if(!Number.isFinite(Number(batchData.chunk_index))||Number(batchData.chunk_index)===0) saveFeatures(batchData);
  }catch(error){
    console.error(`Error saving ${source} batch:`,error);
    return {status:500,body:{success:false,error:'Failed to save batch data'}};
  }
  emitFallDetected(batchData,source);
  io.emit('sensorBatch',batchData);
  return {status:200,body:{success:true,message:`Batch data received via ${source}`,session_id:currentSessionId,label:currentLabel,clients:connectedClients}};
}

function normalizeHttpBatch(body){
  const batchData={...body};
  if(!Number.isFinite(Number(batchData.window_start_ms))&&Array.isArray(batchData.accel_data)&&batchData.accel_data.length>0){
    const firstTimestamp=Number(batchData.accel_data[0].t);
    if(Number.isFinite(firstTimestamp)) batchData.window_start_ms=Math.round(firstTimestamp*1000);
  }
  if(!Number.isFinite(Number(batchData.sample_interval_ms))) batchData.sample_interval_ms=20;
  if(!Number.isFinite(Number(batchData.sample_rate))) batchData.sample_rate=Math.round(1000/Number(batchData.sample_interval_ms||20));
  if(!Number.isFinite(Number(batchData.window_size))&&Array.isArray(batchData.accel_data)) batchData.window_size=batchData.accel_data.length;
  return batchData;
}

function readCoapExtendedValue(nibble,buffer,offset){
  if(nibble<13) return {value:nibble,offset};
  if(nibble===13){
    if(offset>=buffer.length) throw new Error('Truncated CoAP option');
    return {value:buffer[offset]+13,offset:offset+1};
  }
  if(nibble===14){
    if(offset+1>=buffer.length) throw new Error('Truncated CoAP option');
    return {value:buffer.readUInt16BE(offset)+269,offset:offset+2};
  }
  throw new Error('Unsupported CoAP option value');
}

function parseCoapRequest(message){
  if(!Buffer.isBuffer(message)||message.length<4) throw new Error('Invalid CoAP packet');
  const version=message[0]>>6;
  const type=(message[0]>>4)&0x03;
  const tokenLength=message[0]&0x0f;
  if(version!==1) throw new Error(`Unsupported CoAP version: ${version}`);
  if(tokenLength>8||message.length<4+tokenLength) throw new Error('Invalid CoAP token length');
  const code=message[1];
  const messageId=message.readUInt16BE(2);
  const token=message.subarray(4,4+tokenLength);
  let offset=4+tokenLength;
  let currentOptionNumber=0;
  const uriPath=[];
  while(offset<message.length){
    if(message[offset]===0xff){ offset+=1; break; }
    const optionHeader=message[offset++];
    const deltaInfo=readCoapExtendedValue(optionHeader>>4,message,offset);
    const lengthInfo=readCoapExtendedValue(optionHeader&0x0f,message,deltaInfo.offset);
    const optionNumber=currentOptionNumber+deltaInfo.value;
    if(lengthInfo.offset+lengthInfo.value>message.length) throw new Error('Truncated CoAP option payload');
    const optionValue=message.subarray(lengthInfo.offset,lengthInfo.offset+lengthInfo.value);
    offset=lengthInfo.offset+lengthInfo.value;
    currentOptionNumber=optionNumber;
    if(optionNumber===11) uriPath.push(optionValue.toString('utf8'));
  }
  return {type,code,messageId,token,path:`/${uriPath.join('/')}`,payload:offset<=message.length?message.subarray(offset):Buffer.alloc(0)};
}

function buildCoapResponse(request,code,payloadBuffer=null){
  const payload=payloadBuffer&&payloadBuffer.length?payloadBuffer:null;
  const responseType=request.type===COAP_TYPE.CON?COAP_TYPE.ACK:COAP_TYPE.NON;
  const responseMessageId=responseType===COAP_TYPE.ACK?request.messageId:(request.messageId+1)&0xffff;
  const tokenLength=request.token.length;
  const totalLength=4+tokenLength+(payload?payload.length+1:0);
  const response=Buffer.alloc(totalLength);
  response[0]=(1<<6)|(responseType<<4)|tokenLength;
  response[1]=code;
  response.writeUInt16BE(responseMessageId,2);
  request.token.copy(response,4);
  if(payload){
    const markerOffset=4+tokenLength;
    response[markerOffset]=0xff;
    payload.copy(response,markerOffset+1);
  }
  return response;
}

function sendCoapResponse(request,rinfo,code,payloadBuffer=null){
  coap.send(buildCoapResponse(request,code,payloadBuffer),rinfo.port,rinfo.address);
}

function parseRequestedLabel(payload){
  if(!payload||payload.length===0) return null;
  const text=payload.toString('utf8').trim();
  if(text==='0'||text==='1') return text;
  try{
    const parsed=JSON.parse(text);
    const label=String(parsed.label);
    return label==='0'||label==='1'?label:null;
  }catch{
    return null;
  }
}

function decodeSensorBatchPayload(payload){
  if(!Buffer.isBuffer(payload)||payload.length<36) throw new Error('Sensor payload too short');
  let offset=0;
  const version=payload.readUInt8(offset++);
  if(version===LEGACY_SENSOR_PACKET_VERSION){
    const flags=payload.readUInt8(offset++);
    const fsmState=payload.readUInt8(offset++);
    offset+=1;
    const windowSize=payload.readUInt16LE(offset); offset+=2;
    const sampleIntervalMs=payload.readUInt16LE(offset); offset+=2;
    const windowStartMs=payload.readUInt32LE(offset); offset+=4;
    const bpm=payload.readUInt16LE(offset); offset+=2;
    const irRaw=payload.readUInt32LE(offset); offset+=4;
    const features={
      magnitude_avg:payload.readFloatLE(offset),
      sma:payload.readFloatLE(offset+4),
      max_accel:payload.readFloatLE(offset+8),
      max_gyro:payload.readFloatLE(offset+12),
      std_accel:payload.readFloatLE(offset+16),
      jerk_peak:payload.readFloatLE(offset+20)
    };
    offset+=24;
    const expectedLength=offset+windowSize*12;
    if(payload.length!==expectedLength) throw new Error(`Unexpected legacy payload length: got ${payload.length}, expected ${expectedLength}`);
    const accelData=[];
    const gyroData=[];
    for(let i=0;i<windowSize;i+=1){
      const ax=payload.readInt16LE(offset)/400; offset+=2;
      const ay=payload.readInt16LE(offset)/400; offset+=2;
      const az=payload.readInt16LE(offset)/400; offset+=2;
      const gx=payload.readInt16LE(offset)/1000; offset+=2;
      const gy=payload.readInt16LE(offset)/1000; offset+=2;
      const gz=payload.readInt16LE(offset)/1000; offset+=2;
      const timestamp=Math.round(windowStartMs+i*sampleIntervalMs)/1000;
      accelData.push({t:timestamp,x:ax,y:ay,z:az});
      gyroData.push({t:timestamp,x:gx,y:gy,z:gz});
    }
    return {status:'active',bpm,ir_raw:irRaw,window_size:windowSize,sample_rate:Math.round(1000/sampleIntervalMs),sample_interval_ms:sampleIntervalMs,window_start_ms:windowStartMs,fsm_state:fsmState,fall_detected:Boolean(flags&0x01),chunk_index:0,total_chunks:1,features,accel_data:accelData,gyro_data:gyroData};
  }
  if(version!==SENSOR_PACKET_VERSION) throw new Error(`Unsupported sensor packet version: ${version}`);
  const flags=payload.readUInt8(offset++);
  const fsmState=payload.readUInt8(offset++);
  offset+=1;
  const windowSize=payload.readUInt16LE(offset); offset+=2;
  const chunkIndex=payload.readUInt8(offset++);
  const totalChunks=payload.readUInt8(offset++);
  const chunkSamples=payload.readUInt8(offset++);
  offset+=1;
  const sampleIntervalMs=payload.readUInt16LE(offset); offset+=2;
  const windowStartMs=payload.readUInt32LE(offset); offset+=4;
  const bpm=payload.readUInt16LE(offset); offset+=2;
  const irRaw=payload.readUInt32LE(offset); offset+=4;
  const features={
    magnitude_avg:payload.readFloatLE(offset),
    sma:payload.readFloatLE(offset+4),
    max_accel:payload.readFloatLE(offset+8),
    max_gyro:payload.readFloatLE(offset+12),
    std_accel:payload.readFloatLE(offset+16),
    jerk_peak:payload.readFloatLE(offset+20)
  };
  offset+=24;
  if(windowSize<=0||windowSize>200) throw new Error(`Invalid window size: ${windowSize}`);
  if(sampleIntervalMs<=0||sampleIntervalMs>1000) throw new Error(`Invalid sample interval: ${sampleIntervalMs}`);
  if(totalChunks<=0||chunkIndex>=totalChunks) throw new Error(`Invalid chunk info: ${chunkIndex}/${totalChunks}`);
  if(chunkSamples<=0||chunkSamples>windowSize) throw new Error(`Invalid chunk sample count: ${chunkSamples}`);
  const expectedLength=offset+chunkSamples*SENSOR_SAMPLE_BYTES;
  if(payload.length!==expectedLength) throw new Error(`Unexpected payload length: got ${payload.length}, expected ${expectedLength}`);
  const accelData=[];
  const gyroData=[];
  const startIndex=chunkIndex*25;
  for(let i=0;i<chunkSamples;i+=1){
    const timestamp=payload.readFloatLE(offset); offset+=4;
    const ax=payload.readFloatLE(offset); offset+=4;
    const ay=payload.readFloatLE(offset); offset+=4;
    const az=payload.readFloatLE(offset); offset+=4;
    const gx=payload.readFloatLE(offset); offset+=4;
    const gy=payload.readFloatLE(offset); offset+=4;
    const gz=payload.readFloatLE(offset); offset+=4;
    accelData.push({t:timestamp,x:ax,y:ay,z:az});
    gyroData.push({t:timestamp,x:gx,y:gy,z:gz});
  }
  return {status:'active',bpm,ir_raw:irRaw,window_size:windowSize,sample_rate:Math.round(1000/sampleIntervalMs),sample_interval_ms:sampleIntervalMs,window_start_ms:windowStartMs,fsm_state:fsmState,fall_detected:Boolean(flags&0x01),chunk_index:chunkIndex,total_chunks:totalChunks,chunk_start_index:startIndex,features,accel_data:accelData,gyro_data:gyroData};
}

io.on('connection',(socket)=>{
  connectedClients+=1;
  console.log(`✓ Client connected. Total: ${connectedClients}`);
  socket.on('disconnect',()=>{
    connectedClients=Math.max(0,connectedClients-1);
    console.log(`✗ Client disconnected. Total: ${connectedClients}`);
  });
});

app.post('/api/sensor',(req,res)=>{
  const sensorData=req.body;
  if(!sensorData.timestamp) sensorData.timestamp=new Date().toISOString();
  console.log('Received sensor data:',sensorData);
  io.emit('sensorData',sensorData);
  res.json({success:true,message:'Data received and broadcasted',clients:connectedClients});
});

app.post('/api/sensor-batch',(req,res)=>{
  const batchData=normalizeHttpBatch(req.body);
  console.log(`✓ HTTP batch: ${summarizeBatch(batchData)}`);
  const result=processBatchData(batchData,'http');
  res.status(result.status).json(result.body);
});

app.post('/api/session/start',(req,res)=>{
  const previousSessionId=stopCurrentSession('replaced by /api/session/start');
  startNewSession(currentLabel);
  res.json({success:true,previous_session_id:previousSessionId,session_id:currentSessionId,start_time:sessionStartTime,label:currentLabel});
});

app.get('/api/label',(req,res)=>{
  res.json({label:currentLabel,text:getLabelText(currentLabel)});
});

app.post('/api/label',(req,res)=>{
  const {label}=req.body;
  if(label==='0'||label==='1'){
    currentLabel=label;
    const text=getLabelText(label);
    console.log(`🏷️  Label → ${label} (${text})`);
    io.emit('labelChanged',{label:currentLabel,text});
    res.json({success:true,label:currentLabel,text});
    return;
  }
  res.status(400).json({success:false,error:'Invalid label. Use "0" or "1"'});
});

app.post('/api/session/stop',(req,res)=>{
  stopCurrentSession('stopped by HTTP request');
  res.json({success:true});
});

app.post('/api/session/new',(req,res)=>{
  const previousSessionId=stopCurrentSession('replaced by /api/session/new');
  const requestedLabel=req.body&&typeof req.body.label!=='undefined'?String(req.body.label):null;
  const labelToUse=requestedLabel==='0'||requestedLabel==='1'?requestedLabel:currentLabel;
  startNewSession(labelToUse);
  res.json({success:true,previous_session_id:previousSessionId,session_id:currentSessionId,start_time:sessionStartTime,label:labelToUse});
});

app.post('/api/mark-fall',(req,res)=>{
  if(!currentSessionId){
    res.status(400).json({success:false,error:'No active session'});
    return;
  }
  const currentTime=new Date().toISOString();
  const elapsedSeconds=(Date.now()-new Date(sessionStartTime).getTime())/1000;
  fallMarkers.push({timestamp:currentTime,elapsed_seconds:elapsedSeconds,source:'manual'});
  console.log(`🔴 Fall marked at ${elapsedSeconds.toFixed(1)}s`);
  res.json({success:true,elapsed_seconds:elapsedSeconds,marker_count:fallMarkers.length});
});

app.get('/api/sessions/stats',(req,res)=>{
  const countDirs=(dirPath)=>{
    if(!fs.existsSync(dirPath)) return 0;
    return fs.readdirSync(dirPath).filter((name)=>{
      try{ return fs.statSync(path.join(dirPath,name)).isDirectory(); }catch{ return false; }
    }).length;
  };
  res.json({fall:countDirs(FALL_DIR),normal:countDirs(NORMAL_DIR)});
});

app.get('/api/status',(req,res)=>{
  res.json({status:'running',connectedClients,label:currentLabel,session_id:currentSessionId,coap_port:COAP_PORT,timestamp:new Date().toISOString()});
});

coap.on('message',(message,rinfo)=>{
  let request;
  try{
    request=parseCoapRequest(message);
  }catch(error){
    console.error(`CoAP parse error from ${rinfo.address}:${rinfo.port}:`,error.message);
    return;
  }
  if(request.code!==COAP_CODE.POST){
    sendCoapResponse(request,rinfo,COAP_CODE.METHOD_NOT_ALLOWED);
    return;
  }
  switch(request.path){
    case '/api/session/start':
    case '/api/session/new':{
      const previousSessionId=stopCurrentSession('replaced by CoAP session/new');
      const requestedLabel=parseRequestedLabel(request.payload);
      const labelToUse=requestedLabel??currentLabel;
      startNewSession(labelToUse);
      console.log(`✓ CoAP session/new from ${rinfo.address}:${rinfo.port}${previousSessionId?` | previous: ${previousSessionId}`:''}`);
      sendCoapResponse(request,rinfo,COAP_CODE.CREATED);
      break;
    }
    case '/api/session/stop':
      stopCurrentSession('stopped by CoAP request');
      sendCoapResponse(request,rinfo,COAP_CODE.CHANGED);
      break;
    case '/api/sensor-batch':{
      let batchData;
      try{
        batchData=decodeSensorBatchPayload(request.payload);
      }catch(error){
        console.error(`CoAP batch decode error from ${rinfo.address}:${rinfo.port}:`,error.message);
        sendCoapResponse(request,rinfo,COAP_CODE.BAD_REQUEST);
        break;
      }
      console.log(`✓ CoAP batch from ${rinfo.address}:${rinfo.port} | ${summarizeBatch(batchData)}`);
      const result=processBatchData(batchData,'coap');
      if(result.status===200) sendCoapResponse(request,rinfo,COAP_CODE.CHANGED);
      else if(result.status===409) sendCoapResponse(request,rinfo,COAP_CODE.CONFLICT);
      else if(result.status===500) sendCoapResponse(request,rinfo,COAP_CODE.INTERNAL_SERVER_ERROR);
      else sendCoapResponse(request,rinfo,COAP_CODE.BAD_REQUEST);
      break;
    }
    default:
      sendCoapResponse(request,rinfo,COAP_CODE.NOT_FOUND);
      break;
  }
});

coap.on('error',(error)=>{
  console.error('CoAP server error:',error);
});

const localIP=getLocalIP();

server.listen(PORT,'0.0.0.0',()=>{
  console.log('\n╔════════════════════════════════════════╗');
  console.log('║   SENSOR DATA SERVER RUNNING (V3)      ║');
  console.log('╠════════════════════════════════════════╣');
  console.log(`║ HTTP:     http://localhost:${PORT}${' '.repeat(Math.max(0,6-String(PORT).length))} ║`);
  console.log(`║ Network:  http://${localIP}:${PORT}${' '.repeat(Math.max(0,6-localIP.length))} ║`);
  console.log(`║ CoAP:     coap://${localIP}:${COAP_PORT}${' '.repeat(Math.max(0,6-localIP.length))} ║`);
  console.log('╠════════════════════════════════════════╣');
  console.log('║ HTTP status: /api/status               ║');
  console.log('║ CoAP POST:  /api/sensor-batch          ║');
  console.log('║ CoAP POST:  /api/session/new           ║');
  console.log('║ CoAP POST:  /api/session/stop          ║');
  console.log('╠════════════════════════════════════════╣');
  const prefix=process.env.PREFERRED_IP_PREFIX||'auto-detect';
  console.log(`║ IP Prefix: ${prefix}${' '.repeat(Math.max(0,27-prefix.length))} ║`);
  console.log(process.env.SERVER_IP?'║ Mode: Fixed IP (from .env)             ║':'║ Mode: Auto-detect                      ║');
  console.log('╚════════════════════════════════════════╝\n');
  console.log('💡 Đổi IP: Sửa file .env và restart server\n');
});

coap.bind(COAP_PORT,'0.0.0.0',()=>{
  console.log(`✓ CoAP UDP server listening on port ${COAP_PORT}`);
});
