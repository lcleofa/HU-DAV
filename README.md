Student: Gino Cleofa<br>
Periode A (2025/26)<br>
Weekly assignments are added to dedicated feature branch 'hu_dav_lcleofa' in GitHub.

This is the repository for the Master of Applied Data Science course "Data Analysis & Visualisation" at Hogeschool Utrecht.<br>

# Table of Contents
- [Background](#background)
- [Assignments](#assignments)
- [Project structure](#project-structure)
- [Script usage](#script-usage)



# Background
For this course, I am analyzing a WhatsApp group export from my apartment building.<br>
The project focuses on three main categories of discussion:
- Facilities
- Hygiene
- Security

Each category is linked to a specific keyword, which serves as a parameter for the analysis script.<br>
For example, in the Facilities category the keyword can be “lift.”


# Assignments

References python scripts:

| Week | Subject       | Script                        | Image |
|------|--------------------|-------------------------------|-------|
| 2    | Comparing categories| apartment_community_wk2.py    |   wk2_lift_gesprekken_comparing_categories.png    |
| 3    | Time               | apartment_community_wk3.py    |   wk3_beveiliging_camera_time.png    |
| 4    | —                  | apartment_community_wk4.py    |    wk4_log_berichtlengtes_distributions.png   |
| 5    | —                  | apartment_community_wk5.py    |     wk5_aantal_berichten_per_etage_relationship.png  |
| 6    | —                  | apartment_community_wk6.py    |      wk6_pca_modelling_gender_highlighted.png |
| 7    | Final              | apartment_community_final.py  |       |


# Project structure
Below tree depicts relevant project files related to the assignments.<br>
The main. script is `apartment_community.py`<br>
Generated images are saved in the `img` folder.<br>
Logs are saved in the `src` sub folder

```
.
├── README.md
├── checklist.md
├── config.example.toml
├── config.toml
├── dashboards
├── data
├── dev
├── dist
├── img
│   ├── wk2_lift_gesprekken_comparing_categories.png
│   ├── wk3_beveiliging_camera_time.png
│   ├── ...
├── notebooks
│   ├── 01-cleaning.ipynb
│   ├── 02-Gino-comparing_categories.ipynb
...
├── presentations
├── pyproject.toml
├── references
├── src
│   └── wa_analyzer
│       ├── __init__.py
...
│       ├── apartment_community_wk#.py
...
│       ├── logs
│       │   └── logfile.log
...

```

# Script usage
## Install uv package manager
Make sure you have `uv` installed. You can check this by typing `which uv` in the terminal. If that doesnt return a location but `uv not found` you need to install it.<br>
On Unix systems, you can use `curl -LsSf https://astral.sh/uv/install.sh | sh`, for Windows read the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)


## Activate venv
Activate venv<br>
From the root directory of the project activate and verify the `venv`:<br>

```
(base) jdoe-MacBook-Pro:MADS-DAV jdoe$ source .venv/bin/activate
(wa-analyzer) (base) jdoe-MacBook-Pro:MADS-DAV jdoe$
```
## Run script
After this, you can run the scripts with the following `command` and `keyword`, eg:

### week 2
keyword options: ["lift", "schoon", "camera"]<br>
Example
```bash
uv run apartment_community_wk2 --keyword lift
```

### week 3
keyword options: ["lift", "schoon", "camera", "dank"],<br>
Example
```bash
uv run apartment_community_wk3
```

### week 4
```
uv run apartment_community_wk4 --top 5
```

### week 5
```
uv run  apartment_community_wk5 --keyword lift
```

### week 6
```
uv run  apartment_community_wk6 --label Gender --keywords lift camera trap ventilatie
```

## Logs
Inside the `log` folder you will find a logfile, which has some additional information that might be useful for debugging.<br>
For logfile folder location see section 'Project structure'. <br>
The logging is also printed on the terminal output.

## Images
Inside the `img` folder you will find the saved images after each run prefixed by the keyword.<br>
For image folder location see section 'Project structure'. <br>
The images depicts following analyses themes. The images are referenced to the week of the assignments.







