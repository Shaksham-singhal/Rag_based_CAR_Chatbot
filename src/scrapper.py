import json
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException

def setup_driver():
    """Set up and return the Chrome WebDriver with appropriate options."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Uncomment for headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_text_safely(driver, xpath):
    """Extract text from an element safely with error handling."""
    try:
        element = driver.find_element(By.XPATH, xpath)
        return element.text.strip()
    except (NoSuchElementException, StaleElementReferenceException):
        return None
    except Exception as e:
        print(f"Error getting text from {xpath}: {e}")
        return None

def get_car_details(driver):
    """Extract all the car details from the details table."""
    details = {}
    
    try:
        # Look for the details table
        table_xpath = "/html/body/div[2]/div[9]/div[1]/div[2]/div/div[4]/table"
        table = driver.find_element(By.XPATH, table_xpath)
        
        # Find all rows in the table
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        # Process each row except the last rows which have colspan
        for row in rows:
            # Skip rows with colspan (usually footer rows)
            if row.find_elements(By.XPATH, ".//td[@colspan]"):
                continue
                
            # Get cells in the row
            cells = row.find_elements(By.TAG_NAME, "td")
            
            # Make sure we have at least 2 cells for label-value pair
            if len(cells) >= 2:
                label = cells[0].text.strip()
                value = cells[1].text.strip()
                
                # Clean the label (convert to lowercase and remove spaces)
                clean_label = label.lower().replace(' ', '_')
                details[clean_label] = value
                
    except (NoSuchElementException, StaleElementReferenceException):
        print("Car details table not found, trying alternative approaches")
        
        # Try alternative approach - look for table with class 'v_table'
        try:
            table = driver.find_element(By.CSS_SELECTOR, "table.v_table")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                # Skip rows with colspan
                if row.find_elements(By.CSS_SELECTOR, "td[colspan]"):
                    continue
                    
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 2:
                    label = cells[0].text.strip()
                    value = cells[1].text.strip()
                    clean_label = label.lower().replace(' ', '_')
                    details[clean_label] = value
        except Exception as e:
            print(f"Alternative approach for details also failed: {e}")
            
    except Exception as e:
        print(f"Error getting car details: {e}")
    
    return details

def get_card_urls_from_page(driver, page_url):
    """Extract all car card URLs from the listing page."""
    urls = []
    
    try:
        # Navigate to the page
        driver.get(page_url)
        
        # Wait for the page to load and cards to be available
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/div[6]/div[2]/div[1]/div[4]/ul/li[1]"))
        )
        
        # Find all card elements
        card_elements = driver.find_elements(By.XPATH, "/html/body/div[6]/div[2]/div[1]/div[4]/ul/li")
        print(f"Found {len(card_elements)} card elements on the page")
        
        # Extract URLs from each card
        for i in range(1, len(card_elements) + 1):
            try:
                # Get the href attribute from the anchor tag
                card_anchor_xpath = f"/html/body/div[6]/div[2]/div[1]/div[4]/ul/li[{i}]/a"
                card_anchor = driver.find_element(By.XPATH, card_anchor_xpath)
                url = card_anchor.get_attribute('href')
                
                if url:
                    urls.append(url)
                    print(f"Card {i} URL: {url}")
            except Exception as e:
                print(f"Error getting URL for card {i}: {e}")
                continue
                
        return urls
        
    except Exception as e:
        print(f"Error extracting URLs from page {page_url}: {e}")
        return []

def scrape_car_listing(start_page=2, max_pages=50):
    """Scrape car listings from multiple pages."""
    driver = setup_driver()
    all_cars = []
    current_page = start_page
    
    try:
        while current_page <= max_pages:
            # Construct the page URL
            page_url = f"https://www.cartrade.com/second-hand/delhi/page-{current_page}/#so=-1&sc=-1&city=10"
            print(f"Processing page {current_page}: {page_url}")
            
            # Get all card URLs from the current page
            car_urls = get_card_urls_from_page(driver, page_url)
            
            if not car_urls:
                print(f"No car URLs found on page {current_page}. Ending scraping.")
                break
            
            # Process each car URL
            for index, car_url in enumerate(car_urls, 1):
                try:
                    print(f"Processing car {index} of {len(car_urls)} on page {current_page}")
                    print(f"URL: {car_url}")
                    
                    # Navigate to the car details page
                    driver.get(car_url)
                    
                    # Wait for the car details page to load
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.XPATH, "//body"))
                    )
                    
                    # Allow some time for all elements to load
                    time.sleep(2)
                    
                    # Extract car information
                    car_data = {}
                    
                    # Get price
                    car_data['price'] = get_text_safely(driver, "/html/body/div[2]/div[9]/div[1]/div[2]/div/div[1]/div[1]")
                    
                    # Get car name/model
                    car_data['car_name'] = get_text_safely(driver, "/html/body/div[2]/div[9]/div[1]/div[2]/div/div[2]/h1")
                    
                    # Get detailed car specifications
                    car_data['details'] = get_car_details(driver)
                    
                    # Get seller remarks
                    car_data['seller_remarks'] = get_text_safely(driver, "/html/body/div[2]/div[9]/div[1]/div[1]/div[3]")
                    
                    # Add the URL for reference
                    car_data['url'] = car_url
                    
                    # Add the data to our collection
                    all_cars.append(car_data)
                    print(f"Successfully scraped data for {car_data.get('car_name', 'unnamed car')}")
                    
                    # Save partial results after every 10 cars
                    if len(all_cars) % 10 == 0:
                        with open('cartrade_cars_partial.json', 'w', encoding='utf-8') as f:
                            json.dump(all_cars, f, ensure_ascii=False, indent=4)
                        print(f"Saved partial results with {len(all_cars)} cars")
                    
                    # Short delay between requests
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing car {index} on page {current_page}: {e}")
                    continue
            
            # Move to the next page
            current_page += 1
            
            # Save results after each page
            with open('cartrade_cars.json', 'w', encoding='utf-8') as f:
                json.dump(all_cars, f, ensure_ascii=False, indent=4)
            print(f"Saved results after page {current_page-1} with {len(all_cars)} cars")
            
            # Short delay before moving to the next page
            time.sleep(2)
            
    except Exception as e:
        print(f"An error occurred during scraping: {e}")
    finally:
        driver.quit()
        
        # Save the final results
        with open('Project_Car_Chatbot_llm\data\cartrade_cars_final.json', 'w', encoding='utf-8') as f:
            json.dump(all_cars, f, ensure_ascii=False, indent=4)
        print(f"Scraping completed. Saved {len(all_cars)} car listings to Project_Car_Chatbot_llm\data\cartrade_cars_final.json")
    
    return all_cars

if __name__ == "__main__":
    # Start scraping from page 2
    scrape_car_listing(start_page=5, max_pages=500)