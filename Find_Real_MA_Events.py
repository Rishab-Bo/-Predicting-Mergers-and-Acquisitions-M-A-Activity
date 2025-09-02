import pandas as pd
import re
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import warnings
from bs4 import XMLParsedAsHTMLWarning

SEC_EMAIL = 'iib2022004@iiita.ac.in'
USER_AGENT = f'University Project (Contact: {SEC_EMAIL})'
SEC_SLEEP_INTERVAL = 0.2

START_DATE = "2015-01-01" 
SUBMISSIONS_URL_TEMPLATE = 'https://data.sec.gov/submissions/CIK{cik:0>10}.json'
MA_KEYWORD_REGEX = re.compile(r'merger|acquisition|agreement and plan of merger', flags=re.IGNORECASE)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def sec_api_get(url, **kwargs):
    headers = {'User-Agent': USER_AGENT, 'Accept-Encoding': 'gzip, deflate'}
    response = requests.get(url, headers=headers, timeout=30, **kwargs)
    response.raise_for_status()
    time.sleep(SEC_SLEEP_INTERVAL)
    return response

def find_ma_events_in_8k(cik, start_date_str):
    """Scans 8-K filings for a CIK since a given start date to find M&A announcements."""
    submissions_url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    try:
        submissions_data = sec_api_get(submissions_url).json()
    except Exception:
        return []

    recent_filings = submissions_data.get('filings', {}).get('recent', {})
    all_filings_data = [{'form':f, 'filingDate':d, 'accessionNumber':a, 'primaryDocument':p} 
                        for f, d, a, p in zip(recent_filings.get('form', []), recent_filings.get('filingDate', []), recent_filings.get('accessionNumber', []), recent_filings.get('primaryDocument', []))]
    
    found_events = []
    
    for filing in all_filings_data:
        if filing.get('filingDate', '1900-01-01') < start_date_str:
            continue

        if filing.get('form', '') == '8-K':
            acc_no_clean = filing['accessionNumber'].replace('-', '')
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{filing['primaryDocument']}"
            
            try:
                html = sec_api_get(doc_url).content
                soup = BeautifulSoup(html, 'lxml')
                text = soup.get_text(" ", strip=True)

                if "Item 1.01" in text and MA_KEYWORD_REGEX.search(text):
                    event = {'cik': cik, 'event_date': filing['filingDate'], 'filing_url': doc_url}
                    found_events.append(event)
            except Exception:
                continue
                
    return found_events

print("Loading company list...")
companies_df = pd.read_csv("companies_list.csv")
ticker_map_url = 'https://www.sec.gov/files/company_tickers.json'
ticker_data = sec_api_get(ticker_map_url).json()
ticker_to_cik = {row['ticker'].upper(): int(row['cik_str']) for _, row in ticker_data.items()}
CIKS_TO_SCAN = [ticker_to_cik[t.upper()] for t in companies_df['ticker'] if t.upper() in ticker_to_cik]

all_ma_events = []
for cik in tqdm(CIKS_TO_SCAN, desc="Scanning Companies for M&A Events"):
    events = find_ma_events_in_8k(cik, START_DATE)
    if events:
        all_ma_events.extend(events)
        print(f"\nFound {len(events)} event(s) for CIK {cik} since {START_DATE}")

events_df = pd.DataFrame(all_ma_events)
events_df.to_csv('data/real_ma_events.csv', index=False)

print(f"\nScan complete. Found a total of {len(events_df)} potential M&A events since {START_DATE}.")
print("Results saved to 'data/real_ma_events.csv'")