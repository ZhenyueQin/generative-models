import datetime


def get_current_time():
    currentDT = datetime.datetime.now()
    return str(currentDT.strftime("%Y-%m-%d-%H-%M-%S"))



