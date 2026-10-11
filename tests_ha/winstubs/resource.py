"""Windows stand-in for the Linux-only resource module (test use only)."""
RLIMIT_NOFILE = 7
RLIM_INFINITY = -1
def getrlimit(res): return (4096, 4096)
def setrlimit(res, limits): return None
