#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Modules Imported
## ----------------------------
## General (used in scraping)
import time 
import json
import os
import sys

## Specifically scraping
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException 
## ----------------------------


## Set the scraper options
options = Options()
options.add_argument("start-maximized")
options.add_argument("disable-infobars")
options.add_argument("--disable-extensions")
options.headless=True


# Checking for match data
def check_exists_by_xpath(driver, shots):
    """
    Function to check for webpage script elements.

    Args:
        driver (selenium webdriver): webdriver object of link of the webpage to be scrapped.
    
    Returns:
        dict: containing all match data.
    """ 
    if shots:
        try:
            driver.find_element_by_xpath("//script[contains(.,'shotsData')]")
        except NoSuchElementException:
            return False
    else:
        try:
            driver.find_element_by_xpath("//script[contains(.,'matchCentreData')]")
        except NoSuchElementException:
            return False
    return True
    
    
def scrape_understat(link):
        """
        Function to scrape data from the understat-webpage.

        Args:
            link (str): link of the webpage to be scrapped.
    
        Returns:
            dict: containing shot-data.
        """ 
        ## ready the chrome-drivers
        ## here you have to add the path where you have installed drivers for your web-browser
        ## I'm using Chrome --> that's why chromedrivers
        driver = webdriver.Chrome("/Users/Bryce/Projects/selenium/chromedriver")

        ## give the link
        driver.get(link)

        ## get the shot data
        ## Only proceed if it can find match data
        shot_dict = None
        if check_exists_by_xpath(driver, True):
            shot_dict = driver.execute_script("return shotsData;")

        ## close the driver
        driver.close()
        
        return shot_dict


    
    
######
######
# TO DO

## Get all of a team's matches to a csv

# url = "https://www.whoscored.com/Teams/65/Fixtures/Spain-Barcelona"

# elems = driver.find_elements_by_xpath("//a[@href]")
# listofmatches = []
# for elem in elems:
#     strelem = str(elem.get_attribute("href"))
#     if("https://www.whoscored.com/Matches/" in strelem):
#         if("Live" in strelem):
#             listofmatches.append(strelem)
# listofmatches = list(set(listofmatches))
# df_matches = pd.DataFrame(data={"col1": listofmatches})
# df_matches.to_csv("barcamatchlist.csv",encoding='utf-8-sig')
    
# df = pd.read_csv("barcamatchlist.csv",encoding='utf-8-sig')
# urls = df.col1.tolist()
# urls = ['https://www.whoscored.com/Matches/1485383/Live/England-Premier-League-2020-2021-Liverpool-Manchester-United']
    

#########
# Debug #
#########

if __name__ == "__main__":
    print("You ran this module directly (and did not 'import' it).")
    input("\n\nPress the enter key to exit.")