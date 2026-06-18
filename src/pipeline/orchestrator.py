class Pipeline:

    def __init__(self, filters):
        self.filters = filters

    def run(self, scene_path):

        results = {}

        for f in self.filters:

            res = f.run(scene_path)

            # CLEAN NUMPY TYPES HERE
            results[f.name] = {
                "passed": bool(res.passed),
                "reason": res.reason,
                "metrics": {
                    k: (float(v) if hasattr(v, "item") else v)
                    for k, v in res.metrics.items()
                }
            }

            if not res.passed:
                return {
                    "scene": str(scene_path),
                    "accepted": False,
                    "results": results
                }

        return {
            "scene": str(scene_path),
            "accepted": True,
            "results": results
        }