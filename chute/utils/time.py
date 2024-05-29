from datetime import timezone, timedelta, datetime


def get_string_time():
    time = datetime.now(tz=timezone(timedelta(hours=8)))
    return time.strftime("%Y-%m-%d_%H%M%S")


def get_logging_time():
    time = datetime.now(tz=timezone(timedelta(hours=8)))
    return time.strftime("%H:%M:%S")


def get_iso_time():
    time = datetime.now(tz=timezone(timedelta(hours=8)))
    return time.isoformat()
