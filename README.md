Student: Gino Cleofa<br>
Periode: A (2025/26)<br>
Cursus: Data Analysis & Visualisation<br>
Studentnummer: 1908524<br>


This is the repository for the Master of Applied Data Science course "Data Analysis & Visualisation" at Hogeschool Utrecht.<br>

This readme contains the documentation for the weekly assignments.<br>
For each weekly assignments a python script(s) will generate plot(s) for related topic.
This can be depicted in an image file or a streamlit webapp.  

# Table of Contents
- [Background](#background)
- [Assignments](#assignments)
- [Project structure](#project-structure)
- [Script usage](#script-usage)



# Background
For this course, I am analyzing a WhatsApp group export from my apartment building.<br>
The project focuses on following categories:
- Whatsapp activities by authors 
- Facilities

<br>


# Assignments

References python scripts and visualization image files:

| Week | Topic       | Script                        | Image(s) | Webapp|
|------|--------------------|-------------------------------|-------|-------|
| 2    | Comparing categories| apartment_community_wk2.py    |   wk2_emoji_bestuursfunctie_comparing_categories.png <br>  wk2_berichtlengte_bestuursfunctie_comparing_categories.png  | streamlit_dashboard_ac_wk2.py|
| 3    | Time               | apartment_community_wk3.py    |   wk3_beveiliging_camera_time.png    | not applicable|
| 4    | Distributions                  | apartment_community_wk4.py    |    wk4_log_berichtlengtes_distributions.png   | streamlit_dashboard_ac_wk4.py|
| 5    | Relationship                 | apartment_community_wk5.py    |     wk5_aantal_berichten_per_etage_relationship_lift.png  |streamlit_dashboard_ac_wk5.py |
| 6    | Modelling with PCA                 | apartment_community_wk6.py    |      wk6_pca_modelling_gender_highlighted.png | aa|



# Project structure
Below tree depicts relevant project files related to the assignments.<br>
Generated images are saved in the `img` folder.<br>
Generated logs are saved in the `logs` folder.<br>
The `src/wa_analyzer` folder contains the source codes for the source scripts and related web apps.
The `config.toml` contains the references for paths and parameters.
The `pyproject.toml` contains the required packages for the project.

```
.
├── README.md
├── config.toml
├── config
├── data
├── img
│   ├── wk2_emoji_bestuursfunctie_comparing_categories.png
│   ├── wk3_beveiliging_camera_time.png
│   ├── ...
├── logs
├── notebooks
│   ├── 01-cleaning.ipynb
│   ├── 02-Gino-comparing_categories.ipynb
...

├── src
│   └── wa_analyzer
│       ├── __init__.py
...
│       ├── apartment_community_wk#.py
│       ├── streamlit_dashboard_ac_wk#.py
...
├── config.toml
├── pyproject.toml
├── README.md

```


# Script usage
## Install uv package manager
This project uses python package manager 'UV'.
Make sure you have `uv` installed. You can check this by typing `which uv` in the terminal. If that doesnt return a location but `uv not found` you need to install it.<br>
On Unix systems, you can use `curl -LsSf https://astral.sh/uv/install.sh | sh`, for Windows read the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)


## Create venv
Creates your virtual environment and installs all required packages for the project.
```
uv sync --all-extras
```

## Activate venv
From the root directory of the project activate and verify the `venv`:<br>

```
(base) jdoe-MacBook-Pro:MADS-DAV jdoe$ source .venv/bin/activate
(wa-analyzer) (base) jdoe-MacBook-Pro:MADS-DAV jdoe$
```
## Run script
There are 2 options to create the visualizations:
1. run the source script from command line, and view the results as an image file in the `img` folder.
2. run the streamlit webapp from command line, and view the results or adjust variables as desired in the web app.

Both commands should be run from the root directory of the project. In some cases, additional parameters may be required.
The parameter(s) can be found in the 
For some assigments a streamlit webapp is not available.

### week 2
parameter reference: "emoji: number of emojies in the chat per author; length: average message length in the chat per author"<br>

Usage option 1:
```
uv run src/wa_analyzer/apartment_community_wk2.py --metric emoji
uv run src/wa_analyzer/apartment_community_wk2.py --metric length

```

Usage option 2:
```
streamlit run src/wa_analyzer/streamlit_dashboard_ac_wk2.py
```


### week 3
parameter reference:

Usage option 1:
```
uv run src/wa_analyzer/apartment_community_wk3.py

```

### week 4
parameter description: "number of top authors"
parameter options: 10 (default)

Usage option 1:
```
uv run src/wa_analyzer/apartment_community_wk4.py
uv run src/wa_analyzer/apartment_community_wk4.py --top 20
```

Usage option 2:
```
streamlit run src/wa_analyzer/streamlit_dashboard_ac_wk4.py
```

### week 5
parameter description: "keyword related facilities or hygiene"
parameter options: ["lift", "schoon", "camera"]<br>

Usage option 1, example:
```
uv run src/wa_analyzer/apartment_community_wk5.py --keyword lift
```

Usage option 2:
```
streamlit run src/wa_analyzer/streamlit_dashboard_ac_wk5.py
```

### week 6
parameter options in config.toml:

Usage option 1:
```
uv run src/wa_analyzer/apartment_community_wk6.py
```

## Logs
Inside the `log` folder you will find a logfile, which has some additional information that might be useful for debugging.<br>
For logfile folder location see section 'Project structure'. <br>
The logging is also printed on the terminal output.

## Images
Inside the `img` folder you will find the saved images after each run prefixed by the keyword.<br>
For image folder location see section 'Project structure'. <br>
The images depicts following analyses themes. The images are referenced to the week of the assignments.







