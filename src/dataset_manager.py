"""Phase 2: Banking Dataset Manager.

Generates 1,000 realistic, non-sensitive banking text records across 10 specialized domains:
1. Fraud Detection
2. Banking Transactions
3. Credit Cards
4. Loan Policies
5. Customer Authentication
6. Financial Compliance (AML/KYC)
7. Transaction Monitoring
8. Account Management
9. Risk Assessment
10. Cybersecurity

Also generates 10 representative semantic search queries.
Preserves original text, unique integer IDs, domain tags, and metadata.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from config.settings import RAW_DATA_DIR, METADATA_DIR

DOMAINS = [
    "Fraud Detection",
    "Banking Transactions",
    "Credit Cards",
    "Loan Policies",
    "Customer Authentication",
    "Financial Compliance",
    "Transaction Monitoring",
    "Account Management",
    "Risk Assessment",
    "Cybersecurity",
]

# Base templates per domain to generate 100 high-quality variations per domain (total 1,000 records)
DOMAIN_TEMPLATES = {
    "Fraud Detection": [
        "Unusual transaction pattern detected on account ending in {acc}: rapid sequence of micro-payments from foreign IP {ip}.",
        "Automated fraud alert triggered for cardholder {name}: sudden card-not-present attempt exceeding daily threshold of ${amount}.",
        "Geo-velocity anomaly detected: ATM withdrawal initiated in {city_a} within 12 minutes of POS authorization in {city_b}.",
        "Stolen credential check flagged high-risk login attempt originating from known Tor exit node on device ID {dev}.",
        "Account takeover prevention system blocked unauthorized wire transfer request of ${amount} to newly added beneficiary.",
        "Sim-swap suspicion raised: mobile authentication token requested immediately following carrier transfer notification.",
        "Card skimming indicator: multiple declined fallback magstripe swipes reported at terminal {term} in {city_a}.",
        "Synthetic identity profile detected during automated KYC scan: mismatched SSN and DOB combinations on applicant file.",
        "Phishing campaign victim remediation: temporary freeze placed on savings balance following phishing kit report.",
        "Merchant chargeback threshold breach: online electronics merchant {merch} exceeded 2.5% monthly fraud dispute ratio."
    ],
    "Banking Transactions": [
        "Domestic ACH credit transfer of ${amount} successfully settled to account {acc} from payroll provider.",
        "International SWIFT MT103 wire instruction processed for ${amount} EUR equivalent to beneficiary bank in {city_b}.",
        "Recurring standing order executed: monthly mortgage amortization payment of ${amount} debited from checking ledger.",
        "Real-time payment (RTP) network notification: instant credit confirmation received for reference {ref}.",
        "Commercial paper settlement: liquidity transfer of ${amount} credited to corporate operating reserve account.",
        "End-of-day treasury batch sweep completed: excess liquidity moved to automated overnight interest-bearing sweep fund.",
        "Intra-bank ledger transfer of ${amount} completed between primary checking and designated high-yield savings vault.",
        "Cross-border remittance payout completed via local clearing partnership in {city_a} for consumer beneficiary.",
        "Escrow release instruction confirmed: disbursement of ${amount} authorized following title deeds verification.",
        "Merchant settlement batch payout: net POS receipts totaling ${amount} cleared after acquiring interchange fees."
    ],
    "Credit Cards": [
        "Credit limit increase approved for Platinum Mastercard holder: revolving ceiling adjusted from $5,000 to $12,000.",
        "Foreign exchange transaction surcharge waived on international travel purchases for premium sapphire cardholders.",
        "Zero-liability dispute resolution initiated for unauthorized merchant subscription charge of ${amount}.",
        "Cashback reward points redemption: 25,000 accrued points converted into statement credit toward balance.",
        "Virtual card number generated for single-use e-commerce checkout with spending cap fixed at ${amount}.",
        "Annual percentage rate (APR) recalculation: variable purchase interest rate adjusted following prime rate update.",
        "Balance transfer promotional rate locked: 0% introductory APR applied for 18 billing cycles with 3% transfer fee.",
        "Late payment grace period notification: minimum due reminder sent prior to reporting to credit bureaus.",
        "Corporate expense card automated reconciliation: receipt matched against travel per diem booking record.",
        "Contactless EMV tap-to-pay authorization: secure tokenized cryptogram validated for contactless POS terminal."
    ],
    "Loan Policies": [
        "Debt-to-income (DTI) ratio policy guideline: maximum qualifying threshold capped at 43% for conforming residential mortgages.",
        "Commercial real estate loan underwriting requirement: minimum debt service coverage ratio (DSCR) set to 1.25x.",
        "Small Business Administration (SBA 7a) loan eligibility: applicant must demonstrate at least 24 months operating history.",
        "Automotive collateral valuation: loan-to-value (LTV) ceiling established at 110% of vehicle NADA wholesale clean value.",
        "Personal unsecured loan underwriting: minimum FICO credit score requirement established at 660 for prime tier.",
        "Adjustable-rate mortgage (ARM) index transition: benchmark interest rate pegged to 30-day average SOFR index.",
        "Student refinancing hardship forbearance policy: up to 12 months cumulative interest capitalization allowed.",
        "Construction milestone inspection requirement: loan draw requests require certified architect progress sign-off.",
        "Early repayment penalty clause: commercial term loans exempt from prepayment penalties after year three.",
        "Agricultural equipment financing policy: seasonal balloon amortization schedules permitted for farming operations."
    ],
    "Customer Authentication": [
        "Multi-factor authentication (MFA) step-up challenge required for new device registration on consumer mobile app.",
        "FIDO2 biometric Passkey registration verified: client public key securely enrolled on hardware security module.",
        "One-time passcode (OTP) SMS fallback rate-limited following three consecutive invalid entry attempts.",
        "Hardware security token (YubiKey) registered as primary second factor for corporate treasury authorization.",
        "Voice biometric verification confirmed customer identity during telephone banking wire authorization.",
        "Session timeout enforced: web banking portal disconnected after 10 minutes of user inactivity.",
        "Self-service password recovery flow initiated: identity verified via out-of-band push challenge to enrolled smartphone.",
        "OAuth2 access token refreshed: mobile application granted 60-minute scoped token for account read-only endpoints.",
        "Facial liveness detection completed successfully during digital onboarding video selfie verification.",
        "Failed login lockout policy: online banking credentials temporarily locked after 5 consecutive incorrect attempts."
    ],
    "Financial Compliance": [
        "Anti-Money Laundering (AML) suspicious activity report (SAR) filed for structured cash deposits below $10,000.",
        "Office of Foreign Assets Control (OFAC) sanction screening: zero matches identified against SDN lists for wire.",
        "Customer Due Diligence (CDD) rule verification: ultimate beneficial ownership (UBO) verified above 25% ownership.",
        "Politically Exposed Person (PEP) enhanced due diligence (EDD) mandated: annual senior management review scheduled.",
        "Currency Transaction Report (CTR) submitted to FinCEN for aggregate cash deposits exceeding $10,000 threshold.",
        "Foreign Account Tax Compliance Act (FATCA) tax residency declaration W-8BEN validated for non-US account holder.",
        "General Data Protection Regulation (GDPR) data subject access request (DSAR) executed for banking customer.",
        "Bank Secrecy Act (BSA) record retention requirement: transaction log archives maintained for 5-year statutory period.",
        "Know Your Customer (KYC) document re-certification: corporate business license and proof of operating address renewed.",
        "Market abuse surveillance: suspicious proprietary trading order book pattern flagged for wash-trading compliance check."
    ],
    "Transaction Monitoring": [
        "Real-time behavioral baseline: transaction amount exceeds customer 90-day moving average by more than 4 standard deviations.",
        "Structuring detection algorithm triggered: 4 cash deposits of $9,500 completed at distinct branches within 48 hours.",
        "High-risk corridor monitoring: funds routed through designated jurisdictional tax haven flagged for secondary compliance review.",
        "Dormant account reactivation: first outbound funds movement observed on account inactive for 24 months.",
        "Peer-to-peer (P2P) transaction velocity spike: recipient received 35 instant payments from separate accounts in 2 hours.",
        "Crypto asset exchange on-ramp: fiat deposit to designated cryptocurrency platform flagged for source-of-wealth validation.",
        "High-volume merchant refund anomaly: refund volume exceeded total sales by 150% during current settlement period.",
        "Round-trip transaction detection: circular fund transfers between affiliated LLC entities identified by graph analytics.",
        "Shell company heuristic triggered: registered commercial entity with zero payroll overhead moving multi-million dollar invoices.",
        "Payroll processing anomaly: total payroll disbursement exceeded approved credit facility payroll cap."
    ],
    "Account Management": [
        "Joint checking account ownership modified: secondary signatory added following certified legal power of attorney.",
        "Certificate of Deposit (CD) renewal notification: 12-month deposit auto-renewed at prevailing fixed annual yield of 4.50%.",
        "Overdraft protection linkage established: checking overdraft automatically funded from linked money market reserve.",
        "Account closure request processed: remaining balance of ${amount} disbursed via certified cashier check.",
        "Fee schedule update notification: monthly account maintenance fee waived for accounts maintaining $1,500 minimum balance.",
        "Direct deposit routing verification: employer payroll routing number matched to ACH automated clearing house registry.",
        "E-statement enrollment confirmed: paper statement delivery cancelled in favor of secure online PDF archives.",
        "Beneficiary designation update: Transfer-on-Death (TOD) instructions recorded on primary brokerage sweep account.",
        "Stop payment order executed: cashier check number {ref} invalidated upon customer affidavit of loss.",
        "Tiered interest calculation: premium money market balance categorized under tier 3 rate structure."
    ],
    "Risk Assessment": [
        "Expected loss calculation: Probability of Default (PD) multiplied by Loss Given Default (LGD) and Exposure at Default (EAD).",
        "Basel III capital adequacy framework: Tier 1 risk-weighted asset ratio evaluated at 13.8%, exceeding regulatory minimum.",
        "Stress testing scenario: supervisory severely adverse scenario modeling 40% commercial real estate asset devaluation.",
        "Liquidity Coverage Ratio (LCR): high-quality liquid assets (HQLA) evaluated to cover 30-day net cash outflows.",
        "Counterparty credit risk exposure: bilateral netting agreement and credit valuation adjustment (CVA) calculated.",
        "Internal credit rating downgrade: corporate borrower downgraded from BBB+ to BBB- following Q3 earnings release.",
        "Interest rate risk in the banking book (IRRBB): net interest income sensitivity modeled against +200 bps parallel shift.",
        "Operational risk event logging: software system outage resulting in 45-minute core ledger latency recorded in GRC ledger.",
        "Model risk management validation: independent annual review completed for machine learning credit scoring model.",
        "Concentration risk limit monitoring: single industry exposure to commercial office construction capped at 15% of loan book."
    ],
    "Cybersecurity": [
        "Distributed Denial of Service (DDoS) mitigation: scrubbing center successfully repelled 120 Gbps SYN flood attack.",
        "Web Application Firewall (WAF) rule update: SQL injection and remote code execution signatures blocked at gateway.",
        "End-to-end payload encryption: mutual TLS (mTLS) with AES-256-GCM enforced across all microservices and database nodes.",
        "Security Operations Center (SOC) alert: unauthorized privilege escalation attempt detected on domain controller.",
        "Zero Trust network architecture: continuous conditional access verification enforced on internal employee workstations.",
        "Vulnerability scanning report: critical CVE patch applied to core payment gateway API reverse proxies within SLA.",
        "Data Loss Prevention (DLP) safeguard: transmission of unmasked 16-digit credit card primary account numbers blocked.",
        "Incident response dry-run: ransomware recovery scenario tested with air-gapped immutable backup restoration.",
        "Penetration testing finding remediation: cross-site scripting (XSS) parameter vulnerability patched in portal.",
        "Endpoint detection and response (EDR): malicious credential-dumping binary quarantined on employee laptop."
    ],
}

CITIES = ["New York", "London", "Tokyo", "Singapore", "Zurich", "Frankfurt", "Hong Kong", "Dubai", "Sydney", "Toronto"]
NAMES = ["Alice Smith", "Bob Jones", "Carol White", "David Brown", "Emma Wilson", "Frank Miller", "Grace Davis", "Henry Clark"]


def generate_banking_dataset(total_records: int = 1000) -> List[Dict[str, Any]]:
    """Generate 1,000 structured banking records across 10 domains with unique IDs."""
    records_per_domain = total_records // len(DOMAINS)
    dataset = []
    record_id = 0

    for domain in DOMAINS:
        templates = DOMAIN_TEMPLATES[domain]
        for i in range(records_per_domain):
            tmpl = templates[i % len(templates)]
            # Deterministic, reproducible variable injection
            seed_val = record_id + 1000
            acc_num = str(1000000000 + seed_val)[-4:]
            amount = f"{(record_id * 37 % 4500) + 150:.2f}"
            city_a = CITIES[record_id % len(CITIES)]
            city_b = CITIES[(record_id + 3) % len(CITIES)]
            name = NAMES[record_id % len(NAMES)]
            ip = f"192.168.{(record_id % 254) + 1}.{(record_id * 7 % 254) + 1}"
            dev = f"DEV-{record_id:04d}"
            term = f"TERM-{record_id % 50:03d}"
            merch = f"Merchant_{record_id % 80:03d}"
            ref = f"REF-{record_id:06d}"

            text = tmpl.format(
                acc=acc_num,
                amount=amount,
                city_a=city_a,
                city_b=city_b,
                name=name,
                ip=ip,
                dev=dev,
                term=term,
                merch=merch,
                ref=ref
            )

            record = {
                "id": record_id,
                "domain": domain,
                "text": text,
                "metadata": {
                    "account_last4": acc_num,
                    "city": city_a,
                    "record_index": i,
                }
            }
            dataset.append(record)
            record_id += 1

    return dataset


# 10 Representative semantic queries targeting diverse domains
REPRESENTATIVE_QUERIES = [
    {
        "query_id": 0,
        "text": "Detect rapid micro-transactions and unauthorized overseas payment fraud",
        "target_domain": "Fraud Detection",
    },
    {
        "query_id": 1,
        "text": "International wire transfers and real-time ACH payment processing",
        "target_domain": "Banking Transactions",
    },
    {
        "query_id": 2,
        "text": "Credit card credit limit increase and cash back rewards program",
        "target_domain": "Credit Cards",
    },
    {
        "query_id": 3,
        "text": "Residential mortgage debt-to-income underwriting ratio guidelines",
        "target_domain": "Loan Policies",
    },
    {
        "query_id": 4,
        "text": "Two-factor authentication and biometric passkey login security",
        "target_domain": "Customer Authentication",
    },
    {
        "query_id": 5,
        "text": "Anti-money laundering suspicious activity reports and OFAC sanctions compliance",
        "target_domain": "Financial Compliance",
    },
    {
        "query_id": 6,
        "text": "Transaction monitoring algorithms for high velocity structuring and cash deposits",
        "target_domain": "Transaction Monitoring",
    },
    {
        "query_id": 7,
        "text": "Joint bank account signatory authorization and certificate of deposit renewals",
        "target_domain": "Account Management",
    },
    {
        "query_id": 8,
        "text": "Basel III capital adequacy ratios and commercial real estate stress testing",
        "target_domain": "Risk Assessment",
    },
    {
        "query_id": 9,
        "text": "Distributed denial of service attacks and web application firewall security rules",
        "target_domain": "Cybersecurity",
    },
]


def save_banking_dataset(dataset: List[Dict[str, Any]] = None) -> Path:
    """Save dataset to raw JSON and metadata JSONL."""
    if dataset is None:
        dataset = generate_banking_dataset(1000)

    raw_json_path = RAW_DATA_DIR / "banking_dataset.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    metadata_jsonl_path = METADATA_DIR / "records_metadata.jsonl"
    with open(metadata_jsonl_path, "w", encoding="utf-8") as f:
        for r in dataset:
            f.write(json.dumps(r) + "\n")

    queries_path = RAW_DATA_DIR / "representative_queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(REPRESENTATIVE_QUERIES, f, indent=2)

    print(f"[OK] Generated {len(dataset)} banking records across {len(DOMAINS)} domains.")
    print(f"[OK] Raw dataset saved to: {raw_json_path}")
    print(f"[OK] Metadata JSONL saved to: {metadata_jsonl_path}")
    print(f"[OK] Representative queries saved to: {queries_path}")

    return raw_json_path


if __name__ == "__main__":
    save_banking_dataset()
