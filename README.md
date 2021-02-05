# Using data to evaluate soccer games

This repository contains my python work for scraping, analyzing, and understanding soccer matches. It draws data from understat with the match id where uses it to output a number of visualizations and simulate the different possible match outcomes. 

These analyses rely on expected goals, or xG. In layman's term, xG is the **probability** that a shot will result in a goal based on the characteristics of that shot. Such characteristics include things like the shot location, body part of the shot (left foot, right foot, head), the incoming pass, etc. The xG values, by their probabilistic nature can range from 0 - 1, where an xG of `0` is a *certain miss*, while an xG of `1` is one the model thinks is a *certain goal*. An xG of `0.5` would indicate that if identical shots were attempted `10` times, `5` would be expected to result in a goal. 

## Features
* Distributions of players who shoot and their total xG for a match
* Cumulative xG plots for the game flow
* Shot maps (can be drawn for either team individually or both on the same pitch)
* Game result calculator, which uses the two teams' xG to determine probabilities of win/draw/loss for each
* Heatmap show possible game outcomes from their xG
* Histogram of the two teams' potential scores from their xG

## Use
#### Scripts
* The `probability_wdl.py` script contains all of the functions for doing the calculations
* The `soccer_scraper.py` script contains the separate function for scraping the understat data
* The `postmatch_analyzer.ipynb` notebook is set up and ready to run out of the box. The user can enter the understat match id as an input interactively or by assigning the variable to a string. Then, running through the whole notebook will perform the calculations and output the results.

## The Road Ahead
- [ ] Clean this shit up
- [ ] Scrape the event data
- [ ] Passing networks and territory maps
- [ ] Build my own xG model so I don't need to rely on understat
