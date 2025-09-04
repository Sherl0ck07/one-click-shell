# ===== main.py =====

# ===== Environment Config (Suppress Warnings & Logs) =====
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 3 = only fatal
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ===== Standard Library =====
import sys
import ast
import json
import time
import datetime
import logging
logging.getLogger().setLevel(logging.WARNING)  # hide INFO messages

# ===== Third-Party Libraries =====
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== Selenium =====
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# ===== Local Modules =====
from report import generate_html
from score import extract_text_from_pdf, embed
from helpers import scrape_jobs,add_similarity_score

# ===== Paths & Output Setup =====
base_dir = os.path.dirname(os.path.abspath(__file__))

# Create timestamp for outputs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create dedicated output folder per run
output_folder = os.path.join(base_dir, "outputs", f"run_{timestamp}")
os.makedirs(output_folder, exist_ok=True)

# ===== Runner Mode Selection =====
# A = old config, B = new config
run = "B"

# ===== Load Config & Define Filenames =====
if run == "A":
    config_path = os.path.join(base_dir, "config-old.json")
    new_filename = f"job_crawl_summary_A_{timestamp}.html"
    log_file = os.path.join(output_folder, f"output_A_{timestamp}.log")
else:
    config_path = os.path.join(base_dir, "config.json")
    new_filename = f"job_crawl_summary_B_{timestamp}.html"
    log_file = os.path.join(output_folder, f"output_B_{timestamp}.log")
# else:
#     config_path = os.path.join(base_dir, "config - Mayuri.json")
#     new_filename = f"job_crawl_summary_M_{timestamp}.html"
#     log_file = os.path.join(output_folder, f"output_M_{timestamp}.log")

# Load config file
with open(config_path, "r") as f:
    config = json.load(f)

# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8")
    ]
)

logger = logging.getLogger(__name__)

# ===== Extract Config Parameters =====
resume_path = config.get("resume_path")
username = config.get("username")
password = config.get("password")

# Output HTML file path
output_file_path = os.path.join(output_folder, new_filename)

logger.info(f"Resume Path: {resume_path}")
logger.info(f"Username: {username}")
logger.info(f"Output File Path: {output_file_path}")

# ===== Setup Selenium Chrome Options =====
options = Options()
options.add_experimental_option("prefs", {
    "credentials_enable_service": False,
    "profile.password_manager_enabled": False,
    "profile.managed_default_content_settings.images": 2
})
options.add_argument("--start-maximized")
options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Suppress DevTools logs
options.add_argument("--log-level=3")  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

# Silence ChromeDriver logs
service = Service(log_path='NUL')  # Windows null device
driver = webdriver.Chrome(service=service, options=options)

# ===== Load Embedding Model =====
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Domain-specific model for resumes & job descriptions
model_ = SentenceTransformer('TechWolf/JobBERT-v2')

# ===== Resume Embedding =====
resume_text = extract_text_from_pdf(resume_path)
resume_embed = embed(model_, resume_text)
logger.info("Resume Extracted")

# ===== Login to Naukri =====
driver.get("https://www.naukri.com/nlogin/login")

WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "usernameField"))).send_keys(username)
WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "passwordField"))).send_keys(password)
WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Login')]"))).click()

logger.info("Login successful!")

# Allow page load
time.sleep(5)

# XPath selectors for navigation
job_links_xpath = "/html/body/div/div/main/div[1]/div[2]/div[2]/div/div/div/div[1]/h2/a"

# ===== Job Scraping =====
global data
data = []

with open("links.txt", "r") as f:
    lk = ast.literal_eval(f.read())
    logger.info(f"Links: {repr(lk)}")

# Scrape jobs from provided links
for l in lk:
    driver.get(l[1])
    scrape_jobs(data, driver, job_links_xpath,l[0])

logger.info(f"Total scraped jobs: {len(data)}")

# Drop duplicates based on "Job Title" and "Company Name"
seen = set()
unique_data = []
for job in data:
    key = (job.get("Job Title"), job.get("Company Name"))
    if key not in seen:
        seen.add(key)
        unique_data.append(job)

data = unique_data

# ===== Resume-JD Similarity Scoring =====
logger.info("Starting similarity scoring...")

total_jobs = len(data)
progress_bar = tqdm(total=total_jobs, desc="Similarity Scoring Progress", unit="job")

# Parallel similarity computation with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:  # Limit workers to avoid GPU memory overload
    futures = {
        executor.submit(add_similarity_score, job, resume_embed, job.get("job_description", ""),model_): job
        for job in data
    }

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error processing job: {e}")
        finally:
            progress_bar.update(1)

logger.info("Similarity scoring completed.")



data = sorted(
    (job for job in data
     if isinstance(job, dict)
     and isinstance(job.get("score"), (float, int))
     and job.get("score") is not None
     and job.get("score") > 0.4),  # threshold
    key=lambda x: x["score"],
    reverse=True
)

# # --- Auto-apply jobs ---
# wait = WebDriverWait(driver, 10)  # 10 seconds wait

# auto_apply_count = 0
# MAX_AUTO_APPLY = 50

# for job in data:
#     if auto_apply_count >= MAX_AUTO_APPLY:
#         print(f"\nReached max auto-apply limit ({MAX_AUTO_APPLY}). Stopping.")
#         break

#     if not job.get("extApp") and job.get("skillMatch"):
#         try:
#             driver.get(job["URL"])

#             # Wait for the "Apply" button to be clickable
#             apply_btn = wait.until(EC.element_to_be_clickable((By.ID, "apply-button")))
#             try:
#                 apply_btn.click()
#             except Exception as click_err:
#                 print(f"Click failed for job {job.get('Job Title', 'N/A')}: {click_err}")
#                 job["auto_apply_status"] = False
#                 job["reason"] = f"Click failed: {click_err}"
#                 continue  # Skip to next job


#             # Wait for either chatbot drawer (failure) or success icon
#             try:
#                 # Wait up to 5 seconds for the failure element (chatbot) first
#                 wait_short = WebDriverWait(driver, 5)
#                 chatbot = wait_short.until(EC.presence_of_element_located((By.CLASS_NAME, "chatbot_DrawerContentWrapper")))
#                 # If we reach here, chatbot appeared → mark as failed
#                 job["auto_apply_status"] = False
#                 job["reason"] = "Blocked by chatbot"
#                 print(f"Auto-apply blocked by chatbot: {job['Job Title']}")
#             except TimeoutException:
#                 # Chatbot did not appear → check for success icon
#                 try:
#                     wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "img[alt='success-icon']")))
#                     job["auto_apply_status"] = True
#                     job["reason"] = "Applied successfully"
#                     auto_apply_count+=1
#                     print(f"Applied successfully: {job['Job Title']}")
#                 except TimeoutException:
#                     job["auto_apply_status"] = False
#                     job["reason"] = "No success icon detected"
#                     print(f"Apply failed: {job['Job Title']}")

#         except (NoSuchElementException, TimeoutException) as e:
#             job["auto_apply_status"] = False
#             job["reason"] = f"Error applying: {str(e)}"
#             print(f"Error applying to job {job['Job Title']}: {str(e)}")

driver.quit()

# ===== Save Data to JSON =====
json_filename = f"job_data_{timestamp}.json"
json_path = os.path.join(output_folder, json_filename)

with open(json_path, "w", encoding="utf-8") as jf:
    json.dump(data, jf, ensure_ascii=False, indent=2)

logger.info(f"Job data JSON saved at {json_path}")


base_dir = os.path.dirname(json_path)
new_filename = os.path.join(base_dir, new_filename)


generate_html(data,new_filename)
