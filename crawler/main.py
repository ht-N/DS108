from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import re

name = "Default"
profile_path = fr"C:/Users/hello/AppData/Local/Microsoft/Edge/User Data"

webdriver_path = fr"C:\htN\UIT\lastyear\helping\main\msedgedriver.exe"
service = Service(webdriver_path)
options = webdriver.EdgeOptions()
options.add_argument(f'--user-data-dir={profile_path}')
options.add_argument(f'--profile-directory={name}')
# options.add_argument("headless")

driver = webdriver.Edge(service=service, options=options)

df = pd.read_csv(fr'C:\htN\UIT\lastyear\helping\main\urls_all.csv')
# df = df.iloc[:1000]


# Truy cập cột 'URL' trong DataFrame
urls = df['URL'].tolist()

job_data = []
processed = 0
total = len(urls)

for url in urls:
    processed += 1
    print(f"Processing {processed}/{total}: {url}")
    
    try:
        driver.get(url)
        
        # Check if captcha appears and wait for manual solving if needed
        try:
            # Wait up to 30 seconds for job details to appear
            print("Waiting for page to load or captcha to be solved...")
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'company-field')]"))
            )
        except:
            # If timeout occurs, ask for manual intervention
            time.sleep(3)
        
        # Extract job information
        job_info = {}
        
        # Get Field
        try:
            field_element = driver.find_element(By.XPATH, "//div[contains(@class, 'company-field')]/div[contains(@class, 'company-value')]")
            job_info['Field'] = field_element.text.strip()
        except:
            job_info['Field'] = "N/A"
            
        # Get Experience
        try:
            exp_element = driver.find_element(By.XPATH, "//div[@id='job-detail-info-experience']//div[contains(@class, 'job-detail__info--section-content-value')]")
            job_info['Experience'] = exp_element.text.strip()
        except:
            job_info['Experience'] = "N/A"
            
        # Get Location
        try:
            location_element = driver.find_element(By.XPATH, "(//div[contains(@class, 'job-detail__info--section')]/div[contains(@class, 'job-detail__info--section-content-value')])[2]")
            job_info['Location'] = location_element.text.strip()
        except:
            job_info['Location'] = "N/A"
            
        # Get Company Size
        try:
            size_element = driver.find_element(By.XPATH, "//div[contains(@class, 'company-scale')]/div[contains(@class, 'company-value')]")
            job_info['Company Size'] = size_element.text.strip()
        except:
            job_info['Company Size'] = "N/A"
            
        # Get Salary
        try:
            salary_element = driver.find_element(By.XPATH, "//div[contains(@class, 'job-detail__info--section-content')]/div[contains(@class, 'job-detail__info--section-content-value')]")
            job_info['Salary'] = salary_element.text.strip()
        except:
            job_info['Salary'] = "N/A"
        
        # DIRECTLY target the job requirements section only
        requirements_section = "N/A"
        
        # Approach 1: Try to find the specific div containing "Yêu cầu ứng viên" header
        try:
            # Look for h3 elements that contain "Yêu cầu ứng viên" text
            req_headers = driver.find_elements(By.XPATH, "//h3[contains(text(), 'Yêu cầu ứng viên')]")
            
            for header in req_headers:
                # Go up to the parent element (likely the job-description__item div)
                parent_div = header.find_element(By.XPATH, "./..")
                
                # Get all text content
                full_text = parent_div.text.strip()
                
                # Extract only the part from "Yêu cầu ứng viên" to "Quyền lợi" if exists
                req_section_start = full_text.find("Yêu cầu ứng viên")
                
                if req_section_start != -1:
                    # Look for the "Quyền lợi" section that might follow
                    benefits_start = full_text.find("Quyền lợi", req_section_start)
                    
                    if benefits_start != -1:
                        # Extract text between "Yêu cầu ứng viên" and "Quyền lợi"
                        requirements_section = full_text[req_section_start:benefits_start].strip()
                    else:
                        # If no "Quyền lợi" section, take everything from "Yêu cầu ứng viên"
                        requirements_section = full_text[req_section_start:].strip()
                        
                        # Check for other potential section headers that might follow
                        other_sections = ["Địa điểm làm việc", "Thời gian làm việc", 
                                         "Cách thức ứng tuyển", "Hạn nộp hồ sơ"]
                        
                        for section in other_sections:
                            section_start = requirements_section.find(section)
                            if section_start != -1:
                                requirements_section = requirements_section[:section_start].strip()
                
                # If we've found and processed a requirements section, stop looking
                if requirements_section != "N/A":
                    break
        
        except Exception as e:
            print(f"Error in approach 1 for getting job requirements: {str(e)}")
        
        # Approach 2: If approach 1 failed, try searching for specific list items in the job detail information
        if requirements_section == "N/A":
            try:
                job_details_box = driver.find_element(By.XPATH, "//div[@class='job-detail__information-detail' and @id='box-job-information-detail']")
                
                # Get all text and try to extract the requirements section using string manipulation
                full_text = job_details_box.text.strip()
                req_start = full_text.find("Yêu cầu ứng viên")
                
                if req_start != -1:
                    # Find the next section that might follow
                    next_sections = ["Quyền lợi", "Địa điểm làm việc", "Thời gian làm việc", 
                                     "Cách thức ứng tuyển", "Hạn nộp hồ sơ"]
                    
                    next_section_pos = len(full_text)
                    for section in next_sections:
                        pos = full_text.find(section, req_start)
                        if pos != -1 and pos < next_section_pos:
                            next_section_pos = pos
                    
                    # Extract requirements section
                    if next_section_pos < len(full_text):
                        requirements_section = full_text[req_start:next_section_pos].strip()
                    else:
                        requirements_section = full_text[req_start:].strip()
            
            except Exception as e:
                print(f"Error in approach 2 for getting job requirements: {str(e)}")
        
        job_info['Job Requirements'] = requirements_section
        
        # Skip storing the full job description to save space
        # job_info['Job Description'] = "N/A"
        
        # Add job URL for reference
        job_info['URL'] = url
        
        job_data.append(job_info)
        print(f"Successfully extracted data from job {processed}")
        
        # Add a small delay between requests to avoid being blocked
        time.sleep(2)
        
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        continue

driver.quit()

# Create DataFrame from the collected data
jobs_df = pd.DataFrame(job_data)

# Save to CSV
jobs_df.to_csv(fr'C:\htN\UIT\lastyear\helping\main\job_details_full.csv', index=False)
print(f"Job details saved, len of jobs: {len(job_data)}")