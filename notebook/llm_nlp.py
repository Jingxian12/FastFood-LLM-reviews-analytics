import os
import json
import pandas as pd
import copy
import time
from google import genai
from google.genai import types
from tqdm import tqdm

# Analyzes fast-food reviews using Gemini API and performs multi-label classification.
def analyze_review_LLM(df, text_column, api_key, batch_size=50, output_csv_path="ground_truth_LLM.csv",verbose=True):
    client = genai.Client(api_key=api_key)

    # Prompt Engineering with few shot prompting
    system_instruction = """
    You are an expert data labeling assistant for fast-food businesses in Malaysia (e.g., McDonald's, KFC, Burger King). 
    The input text contains 1-2 star customer reviews that use English, Bahasa Melayu, Chinese, Rojak, and local slang.
    Label raw data objectively. Ignore offensive language,discriminatory language and focus strictly on extracting operational facts.
    
    Evaluate EACH category independently. Do NOT lazy-label all fields as 0.
    
    ### TARGET CATEGORIES (Return 1 or 0 for EACH):
    - food_quality: Mentions food/drink item condition, preparation, or sensory aspects (taste, temperature, freshness, portion size, burnt, foreign objects like hair/bugs).
      * Includes: Food/drink spilling/leaking due to loose lids or poor packaging that damages the item's final condition.
      * Excludes: Lobby station condiment shortages (e.g., "sos habis kat kaunter"). Must mention sauce condition ON/IN the food item (e.g., "burger kering takde sos").
      * Keywords: "ayam kecik", "burger sejuk", "roti keras", "kedukut sos".
    
    - order_accuracy: Mentions wrong orders, missing paid items, incorrect customizations, or missing baseline condiments/utensils.
      * Strict Rule: Only applies to items paid for and executed on a valid physical receipt.
      * Excludes 1: Missing a discount/promo due to arriving outside official hours or unpaid disputes. 
      * Excludes 2: Food prepared but left sitting on the counter without being handed over (this belongs to staff_professionalism / speed_of_service).
        
    - staff_professionalism: References staff, crew, cashiers, managers, or packers exhibiting poor attitude, face-to-face rudeness, or severe lack of training/competency.
      * Includes: Rudeness ("muka ketat", "kerek"), ignoring customers, refusing to help (e.g., at kiosks, not picking up store phone), serving late orders first, fighting/shouting in front of customers, changing food items without notice, or food being left sitting on the counter ignored despite being ready.
      * Excludes: Macro resource shortage alone (e.g., "tak cukup staff") if existing staff are trying their best.
    
    - speed_of_service: Complaints about long waiting times, slow moving lines, or kitchen/delivery delays.
      * Keywords: "slow gila", "q panjang", "tunggu lama", "lambat" (The word "lambat" automatically triggers this).
        
    - hygiene_cleanliness: Cleanliness issues regarding tables, floors, toilets, trays, store environment, or staff practicing unhygienic food handling (e.g., no gloves).
      * Keywords: "meja kotor", "sticky floor", "sampah penuh", "dirty/kotor" (The words "dirty/kotor" automatically trigger this).
    
    - facility_equipment: Clear issues with physical assets, store premises, infrastructure layout, or actual store operational availability closures.
      * Includes: Broken AC/lights/doors, parking faults, freezing kiosks, layout bottlenecks (e.g., "crowded entrance layout needs design thinking"), or actual failure to adhere to advertised hours (e.g., unexpected physical or delivery channel closures like a completed/attempted order being repeatedly cancelled by the store, or store showing "Tutup" on Foodpanda/Grab during open hours).
      * Excludes 1: Generic supply shortages ("ayam habis") unless explicitly caused by a broken machine (e.g., "ice cream machine rosak").
      * Excludes 2: Customer suggestions or opinions advising the store to close delivery apps (e.g., "Just close Grab if lack of staff" without actual cancellation/closure happening). 

    - product_availability: Core food item, drink, or menu component is actually out of stock or missing from supply chain inventory.
      * Includes: Explicit mentions of items being sold out/finished ("ayam habis", "fries sold out", "nasi lemak tak ada"), or core items being substituted without permission due to shortages.
      * Excludes 1: Unavailability caused strictly by a broken machine (e.g., "fridge rosak" -> facility_equipment:1, product_availability:0).
      * Excludes 2: Self-service lobby station shortages (e.g., empty sauce dispenser pump).
    
    - no_specific_complaint: Triggered ONLY if the review contains absolutely NO specific, actionable store-level operational complaints.
      * Covers: Vague words ("bad", "teruk", "worst shop") without naming objects/staff; only emojis/stars; or macro political/religious commentary (e.g., "boycott for Palestine", "Allahuakbar") that does not critique local store operations.
    
    ### CRITICAL RULES (EVIDENCE-BASED ONLY):
    1. ZERO-ASSUMPTION: If a review contains ONLY ambiguous words (e.g., "kedukut", "kedai kelakar") without explicitly linking it to a food item or staff behavior, set all 7 operational categories to 0 and set no_specific_complaint to 1.
    2. MUTUAL EXCLUSION: If no_specific_complaint is 1, all other 7 categories MUST be 0. If any of the 7 categories is 1, no_specific_complaint MUST be 0.
    3. CO-EXISTENCE: Multiple operational categories CAN be 1 simultaneously if supported by clear text evidence.
    
    ---
    
    ### EXPECTED OUTPUT SCHEMA:
    Every single object in the returned JSON array MUST strictly contain all 8 categories and follow this exact structured template (do not omit any key):
    {
      "review_index": <integer>,
      "food_quality": {"detected": <0 or 1>, "issues": [<string>]},
      "order_accuracy": {"detected": <0 or 1>, "issues": [<string>]},
      "staff_professionalism": {"detected": <0 or 1>, "issues": [<string>]},
      "speed_of_service": {"detected": <0 or 1>, "issues": [<string>]},
      "hygiene_cleanliness": {"detected": <0 or 1>, "issues": [<string>]},
      "facility_equipment": {"detected": <0 or 1>, "issues": [<string>]},
      "product_availability": {"detected": <0 or 1>, "issues": [<string>]},
      "no_specific_complaint": {"detected": <0 or 1>}
    }

    ### COMPACT EXAMPLES FOR LOGIC REFERENCE:
    (Note: For space-saving inside these few-shot prompt examples, only the categories that are triggered are explicitly listed. However, your ACTUAL output must strictly adhere to the full SCHEMA above for every review, initializing untriggered fields with 0 and empty arrays []).

    
    Example 1: 
    Input: "Always making mistakes on orders. Today I received the wrong drink and no condiments included in my take away. Poorly trained staff."
    Output: {
      "order_accuracy": {"detected": 1, "issues": ["wrong drink", "missing condiments"]},
      "staff_professionalism": {"detected": 1, "issues": ["poorly trained staff"]}
    }

    Example 2:
    Input: "Never come to this McD drive thru. Always have to wait for a very long time. Reason for the long wait was technical issue.  Come on… there’s not consistency and reliability in drive thru service. Might as well close the drive thru service. I ended up driving away without ordering. So NEVER use the drive thru for this McD"
    Output: {
      "speed_of_service": {"detected": 1, "issues": ["long wait at drive thru"]},
      "facility_equipment": {"detected": 1, "issues": ["drive thru technical issue"]}
    }

    Example 3:
    Input: "明明24小时，可是喜欢喜欢提早关！"
    Output: {
      "facility_equipment": {"detected": 1, "issues": ["store closed early despite 24h operational hours"]}
    }

    Example 4:
    Input: "ayam selalu habis, burger roti keras...pekerja selalu makan sambil masak.."
    Output: {
      "food_quality": {"detected": 1, "issues": ["hard burger bun"]},
      "staff_professionalism": {"detected": 1, "issues": ["staff eating while cooking"]},
      "product_availability" :{"detected": 1, "issues": ["ayam selau habis"]}
    }

    Example 5:
    Input: "Order kt foodpanda lambat gila service.. certain stuff tk friendly"
    Output: {
      "staff_professionalism": {"detected": 1, "issues": ["unfriendly staff"]},
      "speed_of_service": {"detected": 1, "issues": ["very slow delivery preparation time"]}
    }
    
    Example 6: (Ambiguous single word - No assumptions allowed):
    Input: "kedukut"
    Output: {
      "no_specific_complaint": {"detected": 1}
    }
    
    Example 7 : (Ambiguous word with context - Classification allowed):
    Input: "Ayam kecik gila, kedukut betol branch ni!"
    Output: {
      "food_quality": {"detected": 1, "issues": ["small chicken portion"]}
    }
    
    You MUST respond strictly with a JSON array of objects, keeping the exact same order as the input data.
    
    """
    
    results = []
    start_row = 0
    
    # =========================================================================
    # Initialize CSV Tracking File & Auto-Resume Scanner
    # ========================================================================
    sample_fail_obj = {
        "review_index" : 0,
        "food_quality": {"detected": 0, "issues": []},
        "order_accuracy": {"detected": 0, "issues": []},
        "staff_professionalism": {"detected": 0, "issues": []},
        "speed_of_service": {"detected": 0, "issues": []},
        "hygiene_cleanliness": {"detected": 0, "issues": []},
        "facility_equipment": {"detected": 0, "issues": []},
        "product_availability": {"detected": 0, "issues": []},
        "no_specific_complaint": {"detected": 0},
        "is_api_fail": True
    }
    header_columns = list(df.columns) + list(sample_fail_obj.keys())
    
    # Check for existing progress backup on disk
    if not os.path.exists(output_csv_path):
        pd.DataFrame(columns=header_columns).to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"✨ No existing backup found. Created new CSV tracker at: {output_csv_path}")
    else:
        try:
            existing_df = pd.read_csv(output_csv_path, encoding='utf-8-sig')
            start_row = len(existing_df)
            
            if start_row > 0:
                print(f"⚡ Incomplete task found! {start_row} rows already safely saved on disk.")
                print(f"▶️ Automatically resuming from row {start_row + 1} (DataFrame Index: {start_row})...")
                
                # Load completed labels into memory to construct a unified total output at the end
                results = existing_df[list(sample_fail_obj.keys())].to_dict(orient='records')
            else:
                print(f"Warning: {output_csv_path} exists but it is empty. Starting fresh.")
        except Exception as e:
            print(f"🚨 Failed to parse historical backup ({e}). Starting fresh for safety.")
            start_row = 0

    # If the file on disk already equals or exceeds the runtime df, stop immediately
    if start_row >= len(df):
        print("🎉 Task verified! All rows have already been processed previously. No API calls needed.")
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    
    # =========================================================================
    # Main Processing Loop (Loops starting dynamically from start_row)
    # =========================================================================
    
    # Process the dataframe in batching
    for i in tqdm(range(start_row, len(df), batch_size),desc="LLM Processing"):
        batch_df = df.iloc[i:i+batch_size].copy() # make process data smaller 
        reviews_list = batch_df[text_column].tolist()
        
        # Dynamic Prompt Building
        user_prompt = (
            f"Analyze these reviews sequentially. Output a JSON array with exactly {len(reviews_list)} objects. " # ensure return same amount of review
            f"Each object must strictly contain 'review_index' (integer, matching the Review number below) both 'detected' (integer, strictly use 1 for true and 0 for false) and 'issues' (array of strings) for every category, "
            f"except 'no_specific_complaint' which only needs 'detected'.\n" + 
            "\n".join([f"Review {idx+1}: {text}" for idx, text in enumerate(reviews_list)])
        )
        
        # Safe fallback object for the new nested JSON structure
        default_fail_object = {
            "review_index" : 0,
            "food_quality": {"detected": 0, "issues": []},
            "order_accuracy": {"detected": 0, "issues": []},
            "staff_professionalism": {"detected": 0, "issues": []},
            "speed_of_service": {"detected": 0, "issues": []},
            "hygiene_cleanliness": {"detected": 0, "issues": []},
            "facility_equipment": {"detected": 0, "issues": []},
            "product_availability": {"detected": 0, "issues": []},
            "no_specific_complaint": {"detected": 0},
            "is_api_fail": True # Check whether api is success or not 
        }
        
        batch_final_labels = []
        
        try:
            # 4. Call the Gemini API 
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json", # Force only json format
                    "temperature": 0.0  # Force deterministic evaluation to stop 'lazy' answers
                }
            )
            
            # Parse the text response(json) into a python list
            batch_labels = json.loads(response.text)
            
            # Pre-allocate a list of default failure templates(length of this list strictly matches the actual number of rows in batch_df)
            current_batch_results = [copy.deepcopy(default_fail_object) for _ in range(len(batch_df))]
            
            for item in batch_labels:
                if isinstance(item, dict) and "review_index" in item:
                    try:
                     # E.g., Gemini returns review_index 1, which maps to Python index 0
                        idx = int(item["review_index"]) - 1 
                        
                    # Ensure index returned by Gemini is within safe boundaries of the batch
                        if 0 <= idx < len(current_batch_results):
                            item["is_api_fail"] = False # Successfully parsed, mark as False
                            current_batch_results[idx] = item # Overwrite the default failure object
                    except (ValueError, TypeError):
                        continue # If review_index is not a valid number, skip safely and keep the failure template
            
            batch_final_labels = current_batch_results
            # Append the structured batch results into the master results list
            results.extend(current_batch_results)
            
            # Calculation: With batch_size=30, processing time + a 6-second sleep takes around 18-22 seconds per batch.
            # This limits requests to ~3 times per minute, staying safely below the 15 RPM Tier 1 limit and preventing 429 errors!
            if i + batch_size < len(df): # If it's not the last batch, sleep for a while
                time.sleep(4)
    
            
        except Exception as e:
            print(f"\n[Hard Error] Skipped batch starting at index {i}. Error: {e}")
            error_fill = [copy.deepcopy(default_fail_object) for _ in range(len(batch_df))]
            batch_final_labels = error_fill
            results.extend(error_fill)
            
            # force a 15-second cooldown. This gives the Google servers a proper window to reset the rate counter,
            # preventing the next batch from getting instantly blocked again.
            if i + batch_size < len(df):
                print("Error encountered. Forcing a 15-second cooldown block...")
                time.sleep(15)
        
        # =========================================================================
        # Real-time Append Current Batch to CSV on Disk
        # =========================================================================
        df_batch_labels = pd.DataFrame(batch_final_labels)
        df_batch_labels.index = batch_df.index
        
        batch_output_df = pd.concat([batch_df, df_batch_labels], axis=1)
        
        # Stream rows straight to disk. mode='a' appends, header=False prevents titles corruption
        batch_output_df.to_csv(output_csv_path, mode='a', index=False, header=False, encoding='utf-8-sig')
        
    # =========================================================================
    # Complete Reconstruction & Return Dataframe
    # =========================================================================
    # Convert list output to DataFrame
    df_labels = pd.DataFrame(results)
    
    # Reset indices to guarantee flawless side-by-side concatenation 
    df_clean = df.reset_index(drop=True)
    df_labels = df_labels.reset_index(drop=True)
    
    output_df = pd.concat([df_clean, df_labels], axis=1)
    print(f"\nExecution finished completely! All metrics are safely mirrored at: {output_csv_path}")
    return output_df