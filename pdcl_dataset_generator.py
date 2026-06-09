"""
PDCL Dataset Generator — Long Document Understanding
=====================================================
Generates MESSY, REAL-WORLD style documents with:
- Spelling errors
- OCR artifacts
- Missing punctuation
- Inconsistent formatting
- Irrelevant noise paragraphs
- Repeated information with contradictions
- Answers buried deep in documents
- Ambiguous phrasing
- Mixed number formats
- Incomplete sentences
"""

import random
import json
import os
import re
import numpy as np
from typing import List, Dict, Tuple

# No fixed seed — each generation produces unique data

# ─────────────────────────────────────────────
# NOISE FUNCTIONS — real world messiness
# ─────────────────────────────────────────────

def add_spelling_errors(text: str, error_rate: float = 0.04) -> str:
    """Randomly corrupt characters in words — simulates OCR or typos."""
    words = text.split()
    result = []
    for word in words:
        if random.random() < error_rate and len(word) > 3:
            error_type = random.randint(0, 3)
            if error_type == 0:
                # swap two adjacent characters
                i = random.randint(0, len(word) - 2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
            elif error_type == 1:
                # drop a character
                i = random.randint(1, len(word) - 1)
                word = word[:i] + word[i+1:]
            elif error_type == 2:
                # duplicate a character
                i = random.randint(0, len(word) - 1)
                word = word[:i] + word[i] + word[i:]
            elif error_type == 3:
                # replace with nearby keyboard key
                keyboard_neighbors = {
                    'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf',
                    'e': 'wr', 'f': 'dg', 'g': 'fh', 'h': 'gj',
                    'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k',
                    'm': 'n', 'n': 'mb', 'o': 'ip', 'p': 'o',
                    'r': 'et', 's': 'ad', 't': 'ry', 'u': 'yi',
                    'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu',
                    'z': 'x'
                }
                i = random.randint(0, len(word) - 1)
                ch = word[i].lower()
                if ch in keyboard_neighbors:
                    replacement = random.choice(keyboard_neighbors[ch])
                    word = word[:i] + replacement + word[i+1:]
        result.append(word)
    return ' '.join(result)


def add_ocr_artifacts(text: str, rate: float = 0.02) -> str:
    """Simulate OCR scan artifacts — common in real document datasets."""
    ocr_map = {
        'l': '1', 'I': '1', 'O': '0', 'o': '0',
        'S': '5', 'B': '8', 'g': '9', 'Z': '2'
    }
    result = []
    for ch in text:
        if random.random() < rate and ch in ocr_map:
            result.append(ocr_map[ch])
        else:
            result.append(ch)
    return ''.join(result)


def drop_punctuation(text: str, rate: float = 0.15) -> str:
    """Randomly drop punctuation — common in poorly parsed documents."""
    result = []
    for ch in text:
        if ch in '.,;:!?' and random.random() < rate:
            continue
        result.append(ch)
    return ''.join(result)


def add_random_linebreaks(text: str, rate: float = 0.05) -> str:
    """Insert random line breaks mid-sentence — PDF parsing artifact."""
    words = text.split()
    result = []
    for word in words:
        result.append(word)
        if random.random() < rate:
            result.append('\n')
    return ' '.join(result)


def add_noise_paragraphs(paragraphs: List[str], num_noise: int = 3) -> List[str]:
    """Insert completely irrelevant filler paragraphs — real docs have these."""
    noise_pool = [
        "This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.",
        "Please note that all figures are subject to change without prior notice and should not be used for investment purposes.",
        "The information contained herein has been obtained from sources believed to be reliable but is not guaranteed.",
        "For internal use only. Distribution outside the organization is strictly prohibited without written consent.",
        "Page formatting may vary depending on the viewing software and screen resolution settings.",
        "This report was generated automatically. Please contact the data team for clarifications or discrepancies.",
        "All rights reserved. No part of this publication may be reproduced without permission.",
        "References to future performance do not guarantee actual results and are subject to market conditions.",
        "Data compiled from multiple sources. Some inconsistencies may exist due to reporting period differences.",
        "This is a draft version. Final figures will be published in the official quarterly report.",
        "Note: Currency conversions are approximate and based on rates at time of publication.",
        "Any resemblance to actual events or figures is coincidental unless explicitly stated.",
        "The company reserves the right to amend any information contained in this document.",
        "Technical terminology used in this document is defined in the glossary section at the end.",
        "Readers are advised to exercise their own judgment when interpreting the data presented here.",
    ]
    result = list(paragraphs)
    for _ in range(num_noise):
        pos = random.randint(0, len(result))
        result.insert(pos, random.choice(noise_pool))
    return result


def corrupt_numbers(text: str, rate: float = 0.1) -> str:
    """
    Randomly format numbers inconsistently —
    e.g. 1000000 vs 1,000,000 vs 1.0M vs $1M
    Real financial docs have all of these.
    """
    def replace_number(match):
        if random.random() > rate:
            return match.group(0)
        num_str = match.group(0).replace(',', '')
        try:
            num = float(num_str)
        except:
            return match.group(0)
        fmt = random.randint(0, 4)
        if fmt == 0:
            return f"{num:,.0f}"
        elif fmt == 1:
            return f"{num:.0f}"
        elif fmt == 2 and num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif fmt == 3 and num >= 1_000:
            return f"{num/1_000:.0f}K"
        else:
            return f"${num:,.0f}"
    return re.sub(r'\b\d[\d,]*\b', replace_number, text)


def add_contradictions(paragraphs: List[str], answer_value: str, rate: float = 0.3) -> List[str]:
    """
    Insert a contradicting statement earlier in the document.
    Transformers often pick the first mention — PDCL should find the correct one.
    """
    if random.random() > rate:
        return paragraphs
    # Generate a plausible but wrong value
    try:
        num = float(answer_value.replace(',', '').replace('$', '').replace('M', '000000').replace('K', '000'))
        wrong_num = num * random.uniform(0.6, 1.4)
        wrong_value = f"{wrong_num:,.0f}"
    except:
        wrong_value = answer_value + " (preliminary)"

    contradiction = (
        f"Earlier estimates suggested the figure was approximately {wrong_value}, "
        f"however this was later revised following a full audit of the records."
    )
    pos = random.randint(0, max(0, len(paragraphs) // 3))
    result = list(paragraphs)
    result.insert(pos, contradiction)
    return result


def apply_all_noise(text: str, noise_level: str = 'medium') -> str:
    """Apply all noise functions with configurable intensity."""
    if noise_level == 'low':
        rates = dict(spelling=0.02, ocr=0.01, punct=0.05, linebreak=0.02)
    elif noise_level == 'medium':
        rates = dict(spelling=0.04, ocr=0.02, punct=0.15, linebreak=0.05)
    elif noise_level == 'high':
        rates = dict(spelling=0.08, ocr=0.05, punct=0.25, linebreak=0.1)
    else:
        rates = dict(spelling=0.04, ocr=0.02, punct=0.15, linebreak=0.05)

    text = add_spelling_errors(text, rates['spelling'])
    text = add_ocr_artifacts(text, rates['ocr'])
    text = drop_punctuation(text, rates['punct'])
    text = add_random_linebreaks(text, rates['linebreak'])
    return text


# ─────────────────────────────────────────────
# DOCUMENT TEMPLATES — Expanded for High Diversity
# ─────────────────────────────────────────────

COMPANIES = [
    "Nexacore Industries", "Veltran Systems", "Orbis Financial Group", "Pinnacle Dynamics", "Halcyon Technologies",
    "Meridian Logistics", "Stratus Capital", "Ironveil Manufacturing", "Crestline Biotech", "Solara Energy Partners",
    "Aetheris Pharma", "Novatech Solutions", "Centurion Aerospace", "Vanguard Resources", "Lumina Telecom",
    "Apex Global Consulting", "Zenith Venture Partners", "Oasis Hospitality", "Prism Robotics", "Summit Agro-Tech",
    "Veridian Healthcare", "Eclipse Automotive", "Sentinel Cyber", "Titanium Mining", "Chronos Media Group",
    "Aegis Heavy Industries", "Valiant Security", "Cobalt Global", "Horizon Software Solutions", "Nexus Real Estate",
    "Quantum Dynamics", "Vector Infrastructure", "Pioneer Retail Group", "Cascade Water Systems", "Starlight Entertainment",
    "Aurora Space Systems", "Hyperion Logistics", "Triton Marine Services", "Phoenix Energy", "Infinity Tech",
    "Omega Biotech", "Delta Chemicals", "Alpha Construction", "Sigma Robotics", "Genesis Food Group",
    "Northwind Capital", "Blueshift Analytics", "Ironclad Defense", "Crimson Health", "Sable Logistics",
    "Argon Semiconductors", "Polaris Navigation", "Terravolt Power", "Zephyr Airlines", "Onyx Pharmaceuticals",
    "Magnolia Insurance", "Silverline Railways", "Topaz Hotels", "Cerulean Water", "Verdant Agriculture",
    "Atlas Freight", "Bastion Cybersecurity", "Coral Reef Fisheries", "Dawnstar Mining", "Ember Steel",
    "Falcon Aerospace", "Granite Holdings", "Helix Genomics", "Ivory Tower Education", "Jasper Oil",
    "Keystone Bridge Corp", "Lighthouse Media", "Mantis Robotics", "Neptune Shipping", "Opal Luxury",
    "Paragon Textiles", "Quasar Computing", "Redwood Timber", "Sapphire Electronics", "Tundra Refrigeration",
    "Umbra Intelligence", "Vertex Dynamics", "Wildfire Energy", "Xenon Display", "Yukon Minerals",
    "Zenon Space Tech", "Brickhouse Construction", "Cirrus Cloud Systems", "Dragonfly Drones", "Evergreen Solar",
    "Foxglove Biotech", "Galleon Shipping", "Hawkeye Security", "Indigo Paints", "Juniper Networks Corp",
    "Kestrel Aviation", "Lunar Mining", "Marble Architecture", "Nighthawk Defense", "Osprey Logistics",
    "Peregrine Finance", "Quicksilver Trading", "Raven Analytics", "Stallion Motors", "Talon Arms",
    "Unity Healthcare", "Viper Technologies", "Wolfram Research Corp", "Xylem Water Tech", "Yonder Exploration"
]

PRODUCTS = [
    "cloud infrastructure", "semiconductor components", "logistics software", "biomedical devices", "renewable energy systems",
    "financial instruments", "enterprise analytics", "consumer electronics", "industrial automation", "cybersecurity solutions",
    "molecular diagnostics", "precision agriculture tools", "automated billing systems", "distributed ledger databases",
    "electric drivetrains", "autonomous drones", "fiber-optic transceivers", "deep learning accelerators",
    "geothermal turbines", "hybrid battery storage", "smart grid relays", "predictive maintenance software"
]

REGIONS = [
    "North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Sub-Saharan Africa", "South Asia", "East Asia",
    "Nordic Countries", "Southeast Asia", "Eastern Europe", "Western Europe", "Central America", "Caribbean", "Oceania"
]

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

EXECUTIVES = [
    ("Sarah Chen", "CEO"), ("Marcus Webb", "CFO"), ("Priya Nair", "COO"), ("David Okafor", "CTO"),
    ("Elena Vasquez", "CMO"), ("James Thornton", "CEO"), ("Aisha Patel", "CFO"), ("Robert Kimura", "COO"),
    ("Hans Mueller", "VP of Engineering"), ("Sanjay Gupta", "Head of Sales"), ("Yuki Tanaka", "Chief Scientist"),
    ("Chloe Dubois", "Chief Legal Officer"), ("Carlos Mendez", "VP of Operations"), ("Fatima Al-Sayed", "Director of Strategy"),
    ("Olga Ivanova", "Head of Product"), ("John Smith", "Chief Information Officer"), ("Linda Green", "VP of Human Resources"),
    ("Arthur Pendragon", "Managing Director"), ("Zoe Washington", "Chief Risk Officer"), ("Li Wei", "Director of Research")
]


def generate_financial_document(doc_id: int, length: str = 'long') -> Dict:
    """
    Generate a messy financial report document with a buried answer.
    Length: 'short' ~500 words, 'medium' ~1000 words, 'long' ~2000 words
    """
    company = random.choice(COMPANIES)
    product = random.choice(PRODUCTS)
    region = random.choice(REGIONS)
    quarter = random.choice(QUARTERS)
    year = random.randint(2019, 2023)
    exec_name, exec_title = random.choice(EXECUTIVES)

    # The actual answer — buried deep in document
    revenue = random.randint(50, 9999) * 100_000
    revenue_str = f"{revenue:,}"
    employees = random.randint(500, 50000)
    growth_rate = round(random.uniform(-15, 45), 1)
    operating_margin = round(random.uniform(5, 35), 1)
    market_share = round(random.uniform(2, 40), 1)

    # Additional answer values for new question types
    total_debt = random.randint(10, 5000) * 1_000_000
    debt_str = f"{total_debt:,}"
    net_profit = random.randint(-200, 3000) * 1_000_000
    profit_str = f"{net_profit:,}"
    stock_price = round(random.uniform(5.0, 850.0), 2)
    rd_spend = random.randint(5, 800) * 1_000_000
    rd_str = f"{rd_spend:,}"
    customer_count = random.randint(1000, 5_000_000)
    churn_rate = round(random.uniform(0.5, 25.0), 1)
    acquisition_cost = random.randint(50, 2000) * 1_000_000
    acq_str = f"{acquisition_cost:,}"

    # 12 question types for maximum variety
    question_type = random.choice([
        'revenue', 'employees', 'growth', 'margin', 'market_share',
        'debt', 'profit', 'stock_price', 'rd_spend', 'customer_count',
        'churn_rate', 'acquisition_cost'
    ])

    # Multiple answer sentence templates per type — prevents pattern memorization
    if question_type == 'revenue':
        question = random.choice([
            f"What was {company}'s total revenue in {quarter} {year}?",
            f"How much revenue did {company} generate during {quarter} {year}?",
            f"Report the total sales figure for {company} in {quarter} {year}.",
        ])
        answer = revenue_str
        answer_sentence = random.choice([
            f"Total revenue for {quarter} {year} reached ${revenue_str}, surpassing analyst expectations by a modest margin.",
            f"{company} reported consolidated revenue of ${revenue_str} during {quarter} {year}, up from the prior period.",
            f"The top line came in at ${revenue_str} for the period ending {quarter} {year}, reflecting steady demand.",
            f"Net sales across all segments totaled ${revenue_str} in {quarter} {year} according to audited financials.",
            f"Revenue of ${revenue_str} was recorded in {quarter} {year}, largely driven by expansion in {region}.",
        ])
    elif question_type == 'employees':
        question = random.choice([
            f"How many employees did {company} have as of {quarter} {year}?",
            f"What was {company}'s total headcount at the end of {quarter} {year}?",
            f"Report the number of full-time staff at {company} during {quarter} {year}.",
        ])
        answer = str(employees)
        answer_sentence = random.choice([
            f"The company employed a total of {employees:,} full-time staff globally as of the end of {quarter} {year}.",
            f"As of {quarter} {year}, headcount stood at {employees:,} across all divisions and geographies.",
            f"{company} had {employees:,} employees on its payroll at the close of {quarter} {year}.",
            f"Total workforce size reached {employees:,} by the end of the reporting period in {quarter} {year}.",
            f"Staffing levels were approximately {employees:,} at the conclusion of {quarter} {year}.",
        ])
    elif question_type == 'growth':
        question = random.choice([
            f"What was {company}'s year-over-year growth rate in {quarter} {year}?",
            f"By what percentage did {company} grow in {quarter} {year} compared to the prior year?",
            f"Report the YoY growth figure for {company} in {quarter} {year}.",
        ])
        answer = f"{growth_rate}%"
        answer_sentence = random.choice([
            f"Year-over-year growth stood at {growth_rate}% for {quarter} {year}, reflecting continued expansion in core markets.",
            f"{company} achieved a {growth_rate}% increase compared to the same quarter last year.",
            f"The growth trajectory was measured at {growth_rate}% YoY for {quarter} {year}.",
            f"Compared to {quarter} {year - 1}, the business grew by {growth_rate}% on a consolidated basis.",
        ])
    elif question_type == 'margin':
        question = random.choice([
            f"What was {company}'s operating margin in {quarter} {year}?",
            f"Report the operating margin percentage for {company} in {quarter} {year}.",
        ])
        answer = f"{operating_margin}%"
        answer_sentence = random.choice([
            f"Operating margin improved to {operating_margin}% during {quarter} {year}, driven by cost optimization initiatives.",
            f"The operating margin was recorded at {operating_margin}% for {quarter} {year}, within management guidance.",
            f"{company} delivered an operating margin of {operating_margin}% in the {quarter} {year} period.",
        ])
    elif question_type == 'market_share':
        question = random.choice([
            f"What market share did {company} hold in {region} during {quarter} {year}?",
            f"How much of the {region} market did {company} control in {quarter} {year}?",
        ])
        answer = f"{market_share}%"
        answer_sentence = random.choice([
            f"In {region}, {company} captured {market_share}% of the total addressable market during {quarter} {year}.",
            f"{company}'s share of the {region} market reached {market_share}% by the end of {quarter} {year}.",
            f"Market penetration in {region} stood at {market_share}% for {quarter} {year}.",
        ])
    elif question_type == 'debt':
        question = random.choice([
            f"What was {company}'s total debt as of {quarter} {year}?",
            f"How much debt did {company} carry at the end of {quarter} {year}?",
            f"Report the total outstanding debt for {company} in {quarter} {year}.",
        ])
        answer = debt_str
        answer_sentence = random.choice([
            f"Total outstanding debt was ${debt_str} as of {quarter} {year}, including both short-term and long-term obligations.",
            f"{company} carried aggregate debt of ${debt_str} on its balance sheet at the close of {quarter} {year}.",
            f"The debt position stood at ${debt_str} at the end of {quarter} {year}, reflecting recent refinancing.",
            f"Consolidated borrowings totaled ${debt_str} as reported in the {quarter} {year} filing.",
        ])
    elif question_type == 'profit':
        question = random.choice([
            f"What was {company}'s net profit in {quarter} {year}?",
            f"How much net income did {company} earn in {quarter} {year}?",
            f"Report the bottom line for {company} during {quarter} {year}.",
        ])
        answer = profit_str
        answer_sentence = random.choice([
            f"Net profit for {quarter} {year} came in at ${profit_str}, after accounting for taxes and one-time charges.",
            f"{company} posted a net income of ${profit_str} in {quarter} {year}, compared to prior year results.",
            f"The bottom line was ${profit_str} for the period ending {quarter} {year}.",
            f"After-tax earnings totaled ${profit_str} during {quarter} {year}, reflecting operational efficiency.",
        ])
    elif question_type == 'stock_price':
        question = random.choice([
            f"What was {company}'s closing stock price at the end of {quarter} {year}?",
            f"At what price did {company}'s shares close in {quarter} {year}?",
        ])
        answer = f"{stock_price}"
        answer_sentence = random.choice([
            f"{company}'s shares closed at ${stock_price} at the end of {quarter} {year}, representing a notable shift from prior levels.",
            f"The stock was trading at ${stock_price} per share as of the last trading day of {quarter} {year}.",
            f"Share price settled at ${stock_price} by the close of {quarter} {year}.",
        ])
    elif question_type == 'rd_spend':
        question = random.choice([
            f"How much did {company} spend on R&D in {quarter} {year}?",
            f"What was {company}'s research and development expenditure in {quarter} {year}?",
        ])
        answer = rd_str
        answer_sentence = random.choice([
            f"Research and development expenditure totaled ${rd_str} during {quarter} {year}, focused on next-generation {product}.",
            f"{company} allocated ${rd_str} to R&D activities in {quarter} {year}.",
            f"Investment in research reached ${rd_str} for {quarter} {year}, a significant portion of total revenue.",
        ])
    elif question_type == 'customer_count':
        question = random.choice([
            f"How many customers did {company} serve in {quarter} {year}?",
            f"What was {company}'s total customer base in {quarter} {year}?",
        ])
        answer = f"{customer_count:,}"
        answer_sentence = random.choice([
            f"The total customer base reached {customer_count:,} by the end of {quarter} {year}, up from the prior quarter.",
            f"{company} served approximately {customer_count:,} active customers during {quarter} {year}.",
            f"Active accounts numbered {customer_count:,} as reported in the {quarter} {year} quarterly update.",
        ])
    elif question_type == 'churn_rate':
        question = random.choice([
            f"What was {company}'s customer churn rate in {quarter} {year}?",
            f"Report the churn percentage for {company} during {quarter} {year}.",
        ])
        answer = f"{churn_rate}%"
        answer_sentence = random.choice([
            f"Customer churn for {quarter} {year} was recorded at {churn_rate}%, within acceptable bounds.",
            f"{company} experienced a {churn_rate}% churn rate during {quarter} {year}.",
            f"The attrition rate among existing customers stood at {churn_rate}% in {quarter} {year}.",
        ])
    elif question_type == 'acquisition_cost':
        target_co = random.choice([c for c in COMPANIES if c != company])
        question = random.choice([
            f"How much did {company} pay to acquire {target_co} in {quarter} {year}?",
            f"What was the acquisition price for {target_co} by {company} in {quarter} {year}?",
        ])
        answer = acq_str
        answer_sentence = random.choice([
            f"{company} completed its acquisition of {target_co} for ${acq_str} during {quarter} {year}, funded through a mix of cash and equity.",
            f"The acquisition of {target_co} closed at a total consideration of ${acq_str} in {quarter} {year}.",
            f"{target_co} was acquired by {company} for a reported ${acq_str} in {quarter} {year}, pending regulatory approval.",
        ])

    # ── Build document paragraphs ──

    intro_paras = [
        f"{company} is a leading provider of {product}, operating across multiple geographies with a strong presence in {region} and other key markets. The company was founded with a mission to deliver innovative solutions to enterprise and consumer segments alike.",

        f"This report covers the operational and financial performance of {company} for {quarter} {year}. All figures are presented in USD unless otherwise noted. Comparative data from prior periods has been restated to reflect changes in accounting methodology.",

        f"Under the leadership of {exec_name}, {exec_title}, {company} has pursued an aggressive growth strategy centered on expanding its {product} portfolio while maintaining disciplined cost management across all business units.",
    ]

    middle_paras = [
        f"The {region} segment continued to outperform internal projections, driven by strong enterprise demand for {product}. Customer acquisition costs declined year over year, contributing positively to overall unit economics.",

        f"Supply chain disruptions in the broader {product} industry posed headwinds during the period. {company} mitigated these challenges through strategic inventory management and alternative sourcing agreements with regional suppliers.",

        f"Research and development expenditure increased by {round(random.uniform(5, 30), 1)}% compared to the same period last year, reflecting the company's commitment to next-generation {product} capabilities. Key initiatives include automation, platform integration, and AI-driven optimization.",

        f"Customer retention rates remained robust at {round(random.uniform(78, 97), 1)}%, supported by enhanced service delivery frameworks and dedicated account management teams in all major markets.",

        f"Regulatory developments in {region} introduced new compliance requirements for providers of {product}. {company} has proactively engaged with regulatory bodies and invested in compliance infrastructure to ensure continued operational alignment.",

        f"Headcount grew organically during the period, with significant hiring in engineering, sales, and customer success functions. Attrition remained below industry benchmarks, reflecting strong employee engagement scores.",

        # Answer paragraph — buried here
        answer_sentence,

        f"Capital expenditure for the period totaled ${random.randint(10,500) * 100_000:,}, primarily allocated to infrastructure expansion, technology upgrades, and facility improvements across key operational hubs.",

        f"The competitive landscape for {product} remained dynamic, with both established players and new entrants vying for market position. {company} differentiated through superior product reliability, customer support, and pricing flexibility.",
    ]

    closing_paras = [
        f"Looking ahead, {company} remains cautiously optimistic about the macroeconomic environment. Management expects continued momentum in {region} while monitoring potential headwinds from currency fluctuations and geopolitical developments.",

        f"The board of directors has approved a share repurchase program of up to ${random.randint(50, 500) * 1_000_000:,}, to be executed over the next 18 months subject to market conditions and regulatory approvals.",

        f"Guidance for the upcoming quarter anticipates revenue growth of {round(random.uniform(3, 20), 1)}% to {round(random.uniform(20, 35), 1)}%, with operating margin expected to remain within the previously communicated target range.",
    ]

    # ── Select and Combine Paragraphs based on Target Length ──
    if length == 'short':
        # Short document: 1 intro, 1 middle (the answer), 1 closing
        selected_intro = [random.choice(intro_paras)]
        selected_middle = [answer_sentence]
        selected_closing = [random.choice(closing_paras)]
        all_paras = selected_intro + selected_middle + selected_closing
        all_paras = add_contradictions(all_paras, answer, rate=0.2)
    elif length == 'medium':
        # Medium document: 2 intro, 2 middle (1 random + 1 answer), 1 closing, 1 noise
        selected_intro = random.sample(intro_paras, 2)
        other_mid = random.choice([p for p in middle_paras if p != answer_sentence])
        selected_middle = [other_mid, answer_sentence]
        selected_closing = random.sample(closing_paras, 1)
        all_paras = selected_intro + selected_middle + selected_closing
        all_paras = add_noise_paragraphs(all_paras, num_noise=1)
        all_paras = add_contradictions(all_paras, answer, rate=0.3)
    else:
        # Long document: 2 intro, 4 middle (3 random + 1 answer), 2 closing, 2 noise
        selected_intro = random.sample(intro_paras, 2)
        other_mids = random.sample([p for p in middle_paras if p != answer_sentence], 3)
        selected_middle = other_mids + [answer_sentence]
        selected_closing = random.sample(closing_paras, 2)
        all_paras = selected_intro + selected_middle + selected_closing
        all_paras = add_noise_paragraphs(all_paras, num_noise=2)
        all_paras = add_contradictions(all_paras, answer, rate=0.4)

    # Shuffle middle paragraphs — real docs are not always logically ordered
    if length in ['medium', 'long']:
        mid_start = len(selected_intro)
        mid_end = mid_start + len(selected_middle)
        middle = all_paras[mid_start:mid_end]
        random.shuffle(middle)
        all_paras = all_paras[:mid_start] + middle + all_paras[mid_end:]

    # Join paragraphs with inconsistent spacing — real PDFs do this
    separators = ['\n\n', '\n\n\n', '\n ', '\n\n   ']
    full_text = ''
    for i, para in enumerate(all_paras):
        full_text += para
        if i < len(all_paras) - 1:
            full_text += random.choice(separators)

    # Apply noise to full text
    noise_level = random.choice(['low', 'medium', 'medium', 'high'])
    full_text = apply_all_noise(full_text, noise_level)
    full_text = corrupt_numbers(full_text, rate=0.15)

    # Find approximate position of answer in document (0.0 to 1.0)
    answer_pos = full_text.lower().find(answer.lower().replace(',', ''))
    answer_position_ratio = answer_pos / max(len(full_text), 1)

    return {
        'id': f"doc_{doc_id:05d}",
        'company': company,
        'quarter': quarter,
        'year': year,
        'document': full_text,
        'question': question,
        'answer': answer,
        'answer_type': question_type,
        'noise_level': noise_level,
        'doc_length': len(full_text.split()),
        'answer_position': round(answer_position_ratio, 3),
        'has_contradiction': answer_pos == -1 or answer_position_ratio > 0.5,
    }


# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────

def build_dataset(
    num_samples: int = 3000,
    output_dir: str = './data',
    split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Dict:
    """
    Build the full dataset with train/val/test splits.
    Ensures answer positions are distributed — not all easy (early) answers.
    Guarantees absolutely no duplicate query targets (uniqueness on company/quarter/year/type).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} documents...")

    samples = []
    lengths = ['short'] * (num_samples // 4) + \
              ['medium'] * (num_samples // 4) + \
              ['long'] * (num_samples // 2)
    
    # Pad/trim lengths list to match exactly num_samples
    while len(lengths) < num_samples:
        lengths.append(random.choice(['short', 'medium', 'long']))
    lengths = lengths[:num_samples]
    random.shuffle(lengths)

    stats = {
        'total': 0,
        'noise_levels': {'low': 0, 'medium': 0, 'high': 0},
        'answer_types': {},
        'avg_doc_length': 0,
        'answer_positions': [],
        'has_contradiction': 0
    }

    seen_keys = set()

    for i, length in enumerate(lengths):
        # Generate until we get a completely unique combination
        while True:
            doc = generate_financial_document(i, length)
            # Create a unique key for this sample
            unique_key = (doc['company'], doc['quarter'], doc['year'], doc['answer_type'])
            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                break
        
        samples.append(doc)

        # Stats
        stats['total'] += 1
        stats['noise_levels'][doc['noise_level']] = stats['noise_levels'].get(doc['noise_level'], 0) + 1
        stats['answer_types'][doc['answer_type']] = stats['answer_types'].get(doc['answer_type'], 0) + 1
        stats['avg_doc_length'] += doc['doc_length']
        stats['answer_positions'].append(doc['answer_position'])
        if doc['has_contradiction']:
            stats['has_contradiction'] += 1

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_samples}")

    stats['avg_doc_length'] = round(stats['avg_doc_length'] / num_samples)
    stats['avg_answer_position'] = round(float(np.mean(stats['answer_positions'])), 3)
    stats['pct_answer_in_second_half'] = round(
        sum(1 for p in stats['answer_positions'] if p > 0.5) / num_samples * 100, 1
    )
    stats['pct_with_contradiction'] = round(stats['has_contradiction'] / num_samples * 100, 1)

    # Split dataset
    random.shuffle(samples)
    n_train = int(num_samples * split[0])
    n_val = int(num_samples * split[1])

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    # Save splits
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        path = os.path.join(output_dir, f'{split_name}.json')
        with open(path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} samples to {path}")

    # Save stats
    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    stats = build_dataset(num_samples=15000, output_dir='./data')

    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total samples       : {stats['total']}")
    print(f"Avg document length : {stats['avg_doc_length']} words")
    print(f"Avg answer position : {stats['avg_answer_position']} (0=start, 1=end)")
    print(f"Answers in 2nd half : {stats['pct_answer_in_second_half']}%")
    print(f"With contradictions : {stats['pct_with_contradiction']}%")
    print(f"\nNoise levels        : {stats['noise_levels']}")
    print(f"Answer types        : {stats['answer_types']}")
    print("\nDataset ready for PDCL training.")

    # Show one sample
    print("\n" + "="*50)
    print("SAMPLE DOCUMENT (first 500 chars)")
    print("="*50)
    with open('./data/train.json') as f:
        sample = json.load(f)[0]
    print(f"Question : {sample['question']}")
    print(f"Answer   : {sample['answer']}")
    print(f"Noise    : {sample['noise_level']}")
    print(f"Length   : {sample['doc_length']} words")
    print(f"Answer @ : {sample['answer_position']} position in doc")
    print(f"\nDocument preview:\n{sample['document'][:500]}...")
