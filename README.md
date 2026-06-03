# MCTA4362 Mini Project

## Project Topic

**Design of an Intelligent Controller Using Machine Learning for DC Motor Speed Control**

This project compares a classical **PID controller** with a **Machine Learning-based neural network controller** for controlling the speed of a DC motor.

The selected plant is a DC motor because it is easy to model, easy to simulate in Python, and suitable for showing the difference between classical control and intelligent control.

## Assignment Summary

- Course: MCTA4362 Machine Learning
- Topic: Intelligent controller using machine learning for control applications
- Plant: DC motor
- Control objective: speed tracking and disturbance rejection
- Classical controller: PID controller
- ML controller: neural network controller trained using simulated expert-control data
- Main comparison metrics:
  - Rise time
  - Overshoot
  - Settling time
  - Robustness to disturbance
  - Mean Squared Error (MSE)

> Note: The project brief says the group size is **maximum 4 persons**. The example report template has 5 rows, so the final group size should be confirmed with the lecturer if 5 members are considered.

## Project Structure

```text
.
|-- Mini Project.pdf
|-- Example template report.docx
|-- MCTA4362_DC_Motor_Report_Draft.docx      # generated report draft
|-- README.md
|-- requirements.txt
|-- docs/
|   |-- report_and_presentation_checklist.md
|   |-- report_draft.md
|-- outputs/                                  # generated after running simulation
|   |-- control_signal.svg
|   |-- metrics.csv
|   |-- results_summary.md
|   |-- speed_response.svg
|   |-- time_history.csv
|   |-- training_loss.csv
|   |-- training_loss.svg
|-- src/
|   |-- dc_motor_ml_control.py
|-- tools/
|   |-- build_report_docx.py
```

## How To Run

Install the required package:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python src/dc_motor_ml_control.py
```

Generate or refresh the Word report draft:

```bash
python tools/build_report_docx.py
```

## Expected Output

The simulation creates:

- `outputs/speed_response.svg` - PID vs ML motor speed response
- `outputs/control_signal.svg` - PID vs ML voltage control signal
- `outputs/training_loss.svg` - neural network training loss
- `outputs/metrics.csv` - numeric performance comparison
- `outputs/results_summary.md` - short summary for report writing

## Suggested Report Sections

1. Abstract
2. Introduction
3. System Selection and Modelling
4. PID Controller Design
5. Machine Learning Controller Design
6. Simulation Setup
7. Results and Comparison
8. Discussion
9. Conclusion
10. Individual Contributions

## Group Members

Fill this table before final submission.

| No. | Name | Matric Number | Contribution |
| --- | --- | --- | --- |
| 1 | To be added | To be added | Project coordination, report editing |
| 2 | To be added | To be added | PID controller and simulation |
| 3 | To be added | To be added | ML controller training and analysis |
| 4 | To be added | To be added | Results, presentation, GitHub documentation |

## Submission Reminder

Upload the code and simulation files to GitHub, then submit the GitHub link together with the report through the Google Drive form in the project brief.
