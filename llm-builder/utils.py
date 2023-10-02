
def number_to_words(number):
    
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if 0 <= number < 10:
        return units[number]
    elif 10 <= number < 20:
        return teens[number - 10]
    elif 20 <= number < 100:
        return tens[number // 10] + (" " + units[number % 10] if number % 10 != 0 else "")
    elif 100 <= number < 1000:
        return units[number // 100] + " hundred" + (" and " + number_to_words(number % 100) if number % 100 != 0 else "")
    elif 1000 <= number < 1000000:
        return number_to_words(number // 1000) + " thousand" + (" " + number_to_words(number % 1000) if number % 1000 != 0 else "")
    elif 1000000 <= number < 1000000000:
        return number_to_words(number // 1000000) + " million" + (" " + number_to_words(number % 1000000) if number % 1000000 != 0 else "")
    elif 1000000000 <= number < 1000000000000:
        return number_to_words(number // 1000000000) + " billion" + (" " + number_to_words(number % 1000000000) if number % 1000000000 != 0 else "")
    elif 1000000000000 <= number < 1000000000000000:
        return number_to_words(number // 1000000000000) + " trillion" + (" " + number_to_words(number % 1000000000000) if number % 1000000000000 != 0 else "")
    else:
        raise ValueError("Number out of range")
