class Print:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    RESET = '\033[0m'

    @staticmethod
    def success(message):
        print(f"{Print.GREEN}{message}{Print.RESET}")

    @staticmethod
    def error(message):
        print(f"{Print.RED}Error - {message}{Print.RESET}")

    @staticmethod
    def warning(message):
        print(f"{Print.YELLOW}{message}{Print.RESET}")

    @staticmethod
    def info(message):
        print(f"{Print.BLUE}{message}{Print.RESET}")

    @staticmethod
    def highlight(message):
        print(f"{Print.PURPLE}{message}{Print.RESET}")