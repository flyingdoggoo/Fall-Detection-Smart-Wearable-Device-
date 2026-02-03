# WEDA-FALL Activities Reference

## 🔴 FALLS (F01-F08)

| Code | Description |
|------|-------------|
| F01 | Fall forward while walking caused by a slip |
| F02 | Lateral fall while walking caused by a slip |
| F03 | Fall backward while walking caused by a slip |
| F04 | Fall forward while walking caused by a trip |
| F05 | Fall backward when trying to sit down |
| F06 | Fall forward while sitting, caused by fainting or falling asleep |
| F07 | Fall backward while sitting, caused by fainting or falling asleep |
| F08 | Lateral fall while sitting, caused by fainting or falling asleep |

## 🟢 ADLs - Activities of Daily Life (D01-D11)

| Code | Description |
|------|-------------|
| D01 | Walking |
| D02 | Jogging |
| D03 | Walking up and downstairs |
| D04 | Sitting on a chair, wait a moment, and get up |
| D05 | Sitting a moment, attempt to get up and collapse into a chair |
| D06 | Crouching (bending at the knees), tie shoes, and get up |
| D07 | Stumble while walking |
| D08 | Gently jump without falling (trying to reach high object) |
| D09 | Hit table with hand |
| D10 | Clapping Hands |
| D11 | Opening and closing door |

## 👥 USERS

### Young Participants (U01-U14)
- U01 to U14: Age 20-46, performed all activities

### Elder Participants (U21-U31)
- U21 to U31: Age 77-95
- **Only performed ADLs** (no falls for safety)
- Most performed: D01, D03, D04, D09, D10, D11

## 🔄 RUNS

Each activity was repeated multiple times:
- R01: First trial
- R02: Second trial
- R03: Third trial
- Some activities have more runs

## 📝 Usage in visualize_data.py

```python
# At the top of main():
WEDA_ACTIVITY = 'F02'  # Lateral fall
WEDA_USER = 'U05'      # Young participant 5
WEDA_RUN = 'R02'       # Second trial
```

Then run:
```bash
python visualize_data.py
```
