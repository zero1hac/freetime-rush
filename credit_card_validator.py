def val_credit_card_number(number):
  credit_card_number_rev = number[::-1]
  total = 0
  for i in credit_card_number_rev[1::2]:
  	x = int(i)*2
  	if len(str(x)) == 2:
  		for a in str(x):
  			total += int(a)
  	else:
  		total += int(x)

  for i in credit_card_number_rev[::2]:
  	total += int(i)

  return total


if __name__ == "__main__":
    credit_card_number = raw_input("Enter a credit card number to validate (Mastercard, Visa, Discover, Amex only): ").strip()
    if (int(credit_card_number[:2]) >= 51 and int(credit_card_number[:2]) <= 55 and len(credit_card_number) == 16) or \
    	(int(credit_card_number[0]) == 4 and (len(credit_card_number) == 13 or len(credit_card_number) == 16)) or \
    	((int(credit_card_number[:2]) == 34 or int(credit_card_number[:2]) == 37) and len(credit_card_number) == 15) or \
    	(int(credit_card_number[:4]) == 6011 and len(credit_card_number) == 16):
      if val_credit_card_number(credit_card_number) % 10 == 0:
    	  print "%s is a valid credit card number" % credit_card_number
      else:
    	  print "%s is NOT a valid credit card number" % credit_card_number
    else:
    	print "%s is NOT a valid credit card number" % credit_card_number
