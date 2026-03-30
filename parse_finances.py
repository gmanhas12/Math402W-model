import pdfplumber

def search_pdf(file_path, keywords):
    print(f"\nScanning: {file_path}")
    print("-" * 40)
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                
                if not text:
                    continue
                    
                for keyword in keywords:
                    if keyword.lower() in text.lower():
                        print(f"\n[Page {page_num}] Match for '{keyword}':")
                        
                        # Print the specific lines containing the keyword
                        lines = text.split('\n')
                        for line in lines:
                            if keyword.lower() in line.lower():
                                print(f"  -> {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")

if __name__ == "__main__":
    # Update these file names if they differ on your computer
    fs_pdf = "2024_RainCity_Audited_FS.pdf"
    ar_pdf = "2022-Annual-Report.pdf"

    # Search the audited financials for the total operating expenses
    fs_keywords = ["Statement of Operations", "Total Expenses", "Wages", "Building operations"]
    search_pdf(fs_pdf, fs_keywords)

    # Search the annual report for capacity numbers
    ar_keywords = ["beds", "units", "shelter", "capacity", "housed"]
    search_pdf(ar_pdf, ar_keywords)