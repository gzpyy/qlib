from qlib.contrib.data.handler import Alpha158

class MyAlphaHandler(Alpha158):
    def get_feature_config(self):
        return self.get_custom_config()
    
    @staticmethod
    def get_custom_config():
        conf = {
            "kbar": {},
            "price": {
                "windows": [0, 1, 2, 3, 4,5,6,7,8,9,10],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4,5,6,7,8,9,10], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': [], # rolling operator to use
                #if include is None we will use default operators
                # 'exclude': ['RANK'], # rolling operator not to use
            }
        }
        return MyAlphaHandler.parse_config_to_fields(conf)

    def get_label_config(self):
        return (["Ref($close, -11)/Ref($close, -1) - 1"], ["LABEL0"])