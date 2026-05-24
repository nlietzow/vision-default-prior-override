from functools import cache

import inflect


@cache
def get_inflect_engine():
    return inflect.engine()
