import re
from enum import Enum

class DocumentCategory(Enum):
    TECH = "tech"
    LEGAL = "legal"
    FINANCE = "finance"
    GENERAL = "general"

def categorize_document(content: str) -> DocumentCategory:
    content_lower = content.lower()
    if re.search(r'\b(legal|contract|law|regulation)\b', content_lower):
        return DocumentCategory.LEGAL
    elif re.search(r'\b(finance|money|investment|stock)\b', content_lower):
        return DocumentCategory.FINANCE
    elif re.search(r'\b(tech|code|algorithm|software)\b', content_lower):
        return DocumentCategory.TECH
    else:
        return DocumentCategory.GENERAL