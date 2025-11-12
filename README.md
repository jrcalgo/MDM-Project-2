# MDM-Project-2

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables** (create a `.env` file or export):
   ```bash
   export OPENAI_API_KEY=your_key_here
   export GROQ_API_KEY=your_key_here
   export GOOGLE_KG_API_KEY=your_key_here  # optional
   export TAVILY_API_KEY=your_key_here      # optional
   ```

3. **Run the pipeline:**
   ```bash
   bash main.sh
   ```

Requires Python 3.8+.

---

**MDM (Master Data Management)**

Master data management solution at Honeywell uses third party data providers (Dun and Bradstreet) to enrich customer and vendor data with firmographic information, company family tree (upper hierarchy), executive contacts, industry classification, ESG and 100 more attributes.
MDM Project 2

Honeywell's Master Data Management (MDM) system holds around 5.5 million customer records, but many of these entities are not well-known, similar to how Whole Foods is less known than its parent, Amazon. The primary task is to first locate each company in the universe and then identify its parent and other firmographic information. This entity resolution service is currently provided by Dun & Bradstreet (D&B). However, a key problem is the low confidence results, as 15% to 20% of records are affected by this issue. Honeywell expects high-confidence results from D&B, specifically a confidence score of 0.8 for companies inside the USA and 0.7 for those outside the USA, a standard that is not being consistently met. Therefore, for the records that fall below these confidence thresholds, we want to leverage the use of AI to locate the existence of the company and retrieve the parent information and other firmographic details and compare with the results given by D & B.

**Impact:**

    - Incomplete customer 360 view.
    - Incomplete customer insight for sales and marketing.
    - Impacts the effectiveness of AI.
    - Inability to certify that the company exists.
    - Inability to identify government entities, operational status and more.

**Scope:**

    - Assess Honeywell enterprise master data (customer and vendor) and evaluate Dun &Bradstreet data capability and offerings.
    - Identify gaps in data coverage, data matching, and integration efficacy.
    - Document the methodology behind DNB's data acquisition and ongoing maintenance to ascertain the reliability and legitimacy of DNB data across various subject areas.

