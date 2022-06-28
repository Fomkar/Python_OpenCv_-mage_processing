import cv2
import numpy as np


sbox = (
    (4, 10, 9, 2, 13, 8, 0, 14, 6, 11, 1, 12, 7, 15, 5, 3),
    (14, 11, 4, 12, 6, 13, 15, 10, 2, 3, 8, 1, 0, 7, 5, 9),
    (5, 8, 1, 13, 10, 3, 4, 2, 14, 15, 12, 7, 6, 0, 9, 11),
    (7, 13, 10, 1, 0, 8, 9, 15, 14, 4, 6, 12, 11, 2, 5, 3),
    (6, 12, 7, 1, 5, 15, 13, 8, 4, 10, 9, 14, 0, 3, 11, 2),
    (4, 11, 10, 0, 7, 2, 1, 13, 3, 6, 8, 5, 9, 12, 15, 14),
    (13, 11, 4, 1, 3, 15, 5, 9, 0, 10, 14, 7, 6, 8, 2, 12),
    (1, 15, 13, 0, 5, 7, 10, 4, 9, 2, 3, 14, 6, 11, 8, 12),
)


resim=cv2.imread("d.jpg")
resim_gri=cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
kirp=resim_gri[100:140, 100:140]
x=np.array(kirp, )
newarray = []
x = np.reshape(x, (1, (x.shape[0]*x.shape[1])))
x = x.astype(np.uint8)
print(x.shape[0])
#for i in range(x.shape[0]):
#        for j in range(x.shape[1]):
#           newarray.append(x[i][j])
print(x[0])

def _bit_length(x):
    assert x >= 0
    return len(bin(x)) - 2

def f_function(var, key):
    assert _bit_length(var) <= 32
    assert _bit_length(key) <= 32

    temp = (var + key) % (1 << 32)
    

    output = 0
    for i in range(8):
        output |= ((sbox[i][(temp >> (4 * i)) & 0b1111]) << (4 * i))

    output = ((output >> (32 - 11)) | (output << 11)) & 0xFFFFFFFF

    return output


def round_encryption(input_left, input_right, round_key):
    output_left = input_right
    output_right = input_left ^ f_function(input_right, round_key)

    return output_left, output_right

def round_decryption(input_left, input_right, round_key):
    output_right = input_left
    output_left = input_right ^ f_function(input_left, round_key)

    return output_left, output_right

class GOST:
    def __init__(self):
        self.master_key = [None] * 8

    def set_key(self, master_key):
        assert _bit_length(master_key) <= 256
        for i in range(8):
            self.master_key[i] = (master_key >> (32 * i)) & 0xFFFFFFFF

        # print 'master_key', [hex(i) for i in self.master_key]

    def encrypt(self, plaintext):
        assert _bit_length(plaintext) <= 64
        text_left = plaintext >> 32
        text_right = plaintext & 0xFFFFFFFF
        # print 'text', hex(text_left), hex(text_right)

        for i in range(24):
            text_left, text_right = round_encryption(
                text_left, text_right, self.master_key[i % 8])

        for i in range(8):
            text_left, text_right = round_encryption(
                text_left, text_right, self.master_key[7 - i])

        return (text_left << 32) | text_right
    
    
    def decrypt(self, ciphertext):
        assert _bit_length(ciphertext) <= 64
        text_left = ciphertext >> 32
        text_right = ciphertext & 0xFFFFFFFF

        for i in range(8):
            text_left, text_right = round_decryption(
             text_left, text_right, self.master_key[i])

        for i in range(24):
            text_left, text_right = round_decryption(
             text_left, text_right, self.master_key[(7 - i) % 8])

        return (text_left << 32) | text_right
      
if __name__ == '__main__':     
    key = 0x1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff

    my_GOST = GOST()
    my_GOST.set_key(key)
    
    text = []
    b= []
    c= []
    num = 0
    for i in range(1600):
        pixel = x[0][i]
        ekle = my_GOST.encrypt(pixel)
        text.append(ekle)
        num += 1
    print(text)
    print(num)
    
    # for i in range(len(text)):
    #     tex = my_GOST.decrypt(text[i])
    #     c.append(tex)
    # for i in range(len(c)):
    #     get = (hex(c[i]))
    #     b.append(get)
    # print(b)
   
   
      


cv2.imwrite("d1.jpg", kirp)
