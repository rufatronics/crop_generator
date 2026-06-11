# Crop Disease TLM — Synthetic Data Generator

Automatically generates synthetic Q&A training data for a crop assistant model covering 10 African crops.

## How it works

- Runs every 6 hours via GitHub Actions
- Generates batches of 10 Q&A pairs per Mistral API call
- Pushes to HuggingFace every 1,000 examples
- Fully resumable — state is saved on HuggingFace so it picks up where it stopped
- Target: 500,000 examples total

## Data buckets

| Bucket | % | Purpose |
|--------|---|---------|
| crop_knowledge | 85% | Disease, symptoms, treatment, practices for 10 crops |
| greetings | 10% | Hello, good morning, who are you, small talk |
| out_of_scope | 5% | Politely declining topics outside the 10 crops |

## Crops covered

cassava, cocoa, cowpea, maize, groundnut, mango, plantain, rice, tomato (+ fall armyworm on maize)

## Setup

1. Fork this repo
2. Add secrets in GitHub → Settings → Secrets → Actions:
   - `MISTRAL_API_KEY`
   - `HF_TOKEN`
3. Enable GitHub Actions
4. Trigger manually or wait for scheduled run

## HuggingFace dataset

Data is pushed to: `rufatronics/crop-disease-tlm-data`
