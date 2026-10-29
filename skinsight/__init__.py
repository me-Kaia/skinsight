"""skinsight: 护肤品成分表解读 + 叠用冲突检查 + PyTorch 肤质适配预测。"""
from .kb import KnowledgeBase, Ingredient, ParsedItem, normalize

__version__ = "0.1.0"
__all__ = ["KnowledgeBase", "Ingredient", "ParsedItem", "normalize"]
