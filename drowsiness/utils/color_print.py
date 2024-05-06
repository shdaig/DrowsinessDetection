def _print_green(s): print("\033[92m{}\033[00m".format(s))


def _print_red(s): print("\033[91m{}\033[00m".format(s))


def _print_yellow(s): print("\033[93m{}\033[00m".format(s))


def _print_light_purple(s): print("\033[94m{}\033[00m".format(s))


def _print_purple(s): print("\033[95m{}\033[00m".format(s))


def _print_cyan(s): print("\033[96m{}\033[00m".format(s))


def _print_light_gray(s): print("\033[97m{}\033[00m".format(s))


def printc(s: str, color: str = ''):
    if color == 'g':
        _print_green(s)
    elif color == 'r':
        _print_red(s)
    elif color == 'y':
        _print_yellow(s)
    elif color == 'lp':
        _print_light_purple(s)
    elif color == 'p':
        _print_purple(s)
    elif color == 'cn':
        _print_cyan(s)
    elif color == 'lg':
        _print_light_gray(s)
    else:
        print(s)


def printg(s: str):
    _print_green(s)


def printcn(s: str):
    _print_cyan(s)


def printr(s: str):
    _print_red(s)


def printy(s: str):
    _print_yellow(s)


def printlg(s: str):
    _print_light_gray(s)
