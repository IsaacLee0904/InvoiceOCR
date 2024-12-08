import os
import pandas as pd
from pathlib import Path
from src.ocr_openai import OpenAIVisionProcessor, APIUsageTracker, convert_to_template

def process_all_invoices(data_dir: str, output_dir: str = "output"):
    """處理資料夾中所有發票/收據圖片並產生 CSV 報表"""
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 儲存所有處理結果
    results = []
    
    # 遍歷資料夾中的所有檔案
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                file_path = os.path.join(root, file)
                print(f"\nProcessing: {file_path}")
                
                try:
                    # 處理個別檔案並取得結果
                    processor = OpenAIVisionProcessor()
                    processor.usage_tracker.start()
                    raw_result, invoice_category = processor.process_invoice(file_path)
                    
                    if raw_result:
                        result = convert_to_template(raw_result, invoice_category)
                        metrics = processor.usage_tracker.get_metrics()
                        
                        # 將結果加入列表
                        result_data = result["data"]
                        result_data.update({
                            "file_path": file_path,
                            "execution_time": metrics["execution_time"],
                            "total_cost": metrics["total_cost"]
                        })
                        results.append(result_data)
                            
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
    
    # 建立 DataFrame 並輸出 CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "invoice_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to {csv_path}")
        
        # 計算總體統計
        total_cost = sum(float(r["total_cost"].replace("$", "")) for r in results)
        print(f"\nTotal processed files: {len(results)}")
        print(f"Total cost: ${total_cost:.4f}")
    else:
        print("No results to save")

def main():
    data_dir = "/workspace/data"
    output_dir = "/workspace/output"
    process_all_invoices(data_dir, output_dir)

if __name__ == "__main__":
    main()