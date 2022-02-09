class StringUtils:

    @staticmethod
    def letter_range(start, stop="{", step=1):
        """Yield a range of uppercase letters.""" 
        for ord_ in range(ord(start.upper()), ord(stop.upper()), step):
            yield chr(ord_)