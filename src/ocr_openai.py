# src/ocr_openai.py
import os
import base64
from openai import OpenAI
import json
from typing import Dict, Any
from pathlib import Path
import time
from typing import Dict, Any, Tuple

class APIUsageTracker:
    def __init__(self):
        self.start_time = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = []
    
    def start(self):
        self.start_time = time.time()
    
    def add_api_call(self, name: str, response: Any):
        if hasattr(response, 'usage'):
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            self.api_calls.append({
                'name': name,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            })
    
    def get_metrics(self) -> Dict:
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        
        # Calculate costs based on OpenAI pricing
        input_cost = self.total_input_tokens * (10.0 / 1_000_000)  # $10/1M tokens
        output_cost = self.total_output_tokens * (30.0 / 1_000_000)  # $30/1M tokens
        
        return {
            'execution_time': f"{elapsed_time:.2f} seconds",
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'input_cost': f"${input_cost:.4f}",
            'output_cost': f"${output_cost:.4f}",
            'total_cost': f"${(input_cost + output_cost):.4f}",
            'api_calls': self.api_calls
        }

class OpenAIVisionProcessor:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client with API key"""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.usage_tracker = APIUsageTracker()

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image or PDF to base64 string"""
        try:
            # 檢查文件副檔名
            file_extension = image_path.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                from pdf2image import convert_from_path
                import io
                
                images = convert_from_path(image_path, first_page=1, last_page=1)
                if images:
                    img_byte_arr = io.BytesIO()
                    images[0].save(img_byte_arr, format='JPEG')
                    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            else:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
                    
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return None

    def identify_invoice_type(self, image_path: str) -> str:
        """Identify the type of invoice/receipt using OpenAI Vision"""
        base64_image = self.encode_image_to_base64(image_path)
        
        prompt = """Please analyze this image and identify the exact type of invoice/receipt. 
        Choose from the following categories:
        1. 三聯式發票 (Three-part Invoice)
        2. 二聯式發票 (Two-part Invoice)
        3. 電子發票 (E-Invoice)
        4. 收據 (Receipt)
        5. 收銀機統一發票 (Cash Register Receipt)
        
        Please provide only the category number and name, no additional explanation.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50  
            )
            
            self.usage_tracker.add_api_call('identify_type', response)
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error identifying invoice type: {str(e)}")
            return ""

    def process_invoice(self, image_path: str) -> Dict[Any, Any]:
        """Process invoice image using OpenAI Vision API"""
        invoice_category = self.identify_invoice_type(image_path)
        base64_image = self.encode_image_to_base64(image_path)
        
        prompt = """Please analyze this invoice image and extract the following information in JSON format:
        - invoice_vendor_name (seller's name)
        - invoice_vendor_tax_id (seller's tax ID)
        - invoice_buyer_name (buyer's name)
        - invoice_buyer_tax_id (buyer's tax ID)
        - invoice_amount (total amount)
        - remittance_invoice_date (date)
        - remittance_invoice_no (array of items with amount, product_name)
        - doc_total_amount (total amount)
        
        Please ensure the amounts are numbers without currency symbols and provide dates in YYYY-MM-DD format.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            self.usage_tracker.add_api_call('process_invoice', response)

            # Extract the JSON response
            response_text = response.choices[0].message.content
            try:
                # Try to parse the response as JSON
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, try to extract JSON from the text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(response_text[json_start:json_end])
                else:
                    raise Exception("Failed to parse JSON from response")

            return result, invoice_category

        except Exception as e:
            print(f"Error processing invoice: {str(e)}")
            return None, None

def convert_to_template(raw_result: Dict, invoice_category: str) -> Dict:
    """Convert raw OpenAI result to template format"""
    template = {
        "data": {
            "company_code": "",
            "doc_currency": "",
            "doc_exchange_rate": 0.0,
            "doc_net_amount": 0.0,
            "doc_tax_amount": 0.0,
            "doc_total_amount": 0.0,
            "invoice_amount": "",
            "invoice_buyer_name": "",
            "invoice_buyer_tax_id": "",
            "invoice_category": "",
            "invoice_currency": "TWD",
            "invoice_no": "",
            "invoice_tax_amount": "",
            "invoice_tax_rate": "",
            "invoice_type": invoice_category,
            "invoice_vendor_name": "",
            "invoice_vendor_tax_id": "",
            "office_id": "",
            "remittance_invoice_date": "",
            "remittance_invoice_no": []
        }
    }
    
    # Map existing values from raw result
    data = template["data"]
    
    # Convert all numeric values to strings
    if "doc_total_amount" in raw_result:
        data["doc_total_amount"] = str(raw_result["doc_total_amount"])
    if "invoice_amount" in raw_result:
        data["invoice_amount"] = str(raw_result["invoice_amount"])
    
    # Map string values
    data["invoice_vendor_name"] = raw_result.get("invoice_vendor_name", "")
    data["invoice_vendor_tax_id"] = raw_result.get("invoice_vendor_tax_id", "")
    data["invoice_buyer_name"] = raw_result.get("invoice_buyer_name", "")
    data["invoice_buyer_tax_id"] = raw_result.get("invoice_buyer_tax_id", "") or ""
    data["remittance_invoice_date"] = raw_result.get("remittance_invoice_date", "")
    
    # Process remittance_invoice_no items
    if "remittance_invoice_no" in raw_result:
        for idx, item in enumerate(raw_result["remittance_invoice_no"], 1):
            template_item = {
                "amount": str(item.get("amount", "")),
                "currency": "TWD",
                "po_line": "",
                "po_no": "",
                "product_name": item.get("product_name", ""),
                "quantity": "1",  # Default to 1 if not specified
                "remark": "",
                "sn": str(idx),
                "tax_amount": "",
                "tax_rate": "",
                "unit_price": str(item.get("amount", ""))
            }
            data["remittance_invoice_no"].append(template_item)
    
    return template

def InvoiceProcessor(image_path: str):
    processor = OpenAIVisionProcessor()
    processor.usage_tracker.start()
    
    try:
        raw_result, invoice_category = processor.process_invoice(image_path)
        if raw_result:
            result = convert_to_template(raw_result, invoice_category)
            print("===== OpenAI Vision API Result =====")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # Print usage metrics
            metrics = processor.usage_tracker.get_metrics()
            print("\n===== API Usage Metrics =====")
            print(f"Total Execution Time: {metrics['execution_time']}")
            print(f"Total Input Tokens: {metrics['total_input_tokens']}")
            print(f"Total Output Tokens: {metrics['total_output_tokens']}")
            print(f"Input Cost: {metrics['input_cost']}")
            print(f"Output Cost: {metrics['output_cost']}")
            print(f"Total Cost: {metrics['total_cost']}")
            
            print("\nAPI Calls Breakdown:")
            for call in metrics['api_calls']:
                print(f"- {call['name']}: {call['input_tokens']} input tokens, {call['output_tokens']} output tokens")
        else:
            print("Failed to process invoice")
    except Exception as e:
        print(f"Error in main: {str(e)}")

# if __name__ == "__main__":
#     image_path = "/workspace/data/差旅費\機票\機票01.jpg"
#     main(image_path)