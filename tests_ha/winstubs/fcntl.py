"""Windows stand-in for the Linux-only fcntl module (test use only)."""
LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8
F_GETFL = F_SETFL = F_GETFD = F_SETFD = 0
def flock(fd, op): return None
def lockf(fd, op, *a): return None
def fcntl(fd, op, arg=0): return 0
def ioctl(fd, op, arg=0, mutate=True): return 0
