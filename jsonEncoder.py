import json
import numpy as np

class JsonEncoder(json.JSONEncoder):
    """ Convert numpy class to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            super().default(obj)