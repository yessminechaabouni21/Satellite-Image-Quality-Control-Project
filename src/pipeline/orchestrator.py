class Pipeline:

    def __init__(self, filters):
        self.filters = filters

    def run(self, scene_path):

        results = {}
        failed_filter = None
        failure_reason = None

        for f in self.filters:

            res = f.run(scene_path)

            # CLEAN N               UMPY TYPES HERE
            results[f.name] = {
                "passed": bool(res.passed),
                "reason": res.reason,
                "metrics": {
                    k: (float(v) if hasattr(v, "item") else v)
                    for k, v in res.metrics.items()
                }
            }

            if not res.passed:
                failed_filter = f.name
                failure_reason = res.reason
                return {
                    "scene": str(scene_path),
                    "accepted": False,
                    "failed_filter": failed_filter,
                    "failure_reason": failure_reason,
                    "results": results
                }

        return {
            "scene": str(scene_path),
            "accepted": True,
            "failed_filter": None,
            "failure_reason": None,
            "results": results
        }