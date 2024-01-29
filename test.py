from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os

def generate_aes_key_from_password(password, salt, key_length=32):
    """
    Generate a 256-bit (32-byte) AES key from a given password using scrypt with a fixed salt.
    """
    # Use Scrypt key derivation with the provided salt
    kdf = Scrypt(
        salt=salt,
        length=key_length,
        n=16384,
        r=8,
        p=1,
        backend=default_backend()
    )
    aes_key = kdf.derive(password.encode())
    return aes_key

def encrypt_with_aes_ecb(plaintext, key):
    """
    Encrypt a plaintext string using AES encryption in ECB mode.
    """
    # Configure the Cipher for encryption in ECB mode
    encryptor = Cipher(
        algorithms.AES(key),
        modes.ECB(),
        backend=default_backend()
    ).encryptor()
    
    # Pad the plaintext to be compatible with the block size
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_plaintext = padder.update(plaintext.encode()) + padder.finalize()
    
    # Encrypt
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    return ciphertext

def decrypt_with_aes_ecb(ciphertext, key):
    """
    Decrypt an AES-encrypted ciphertext in ECB mode.
    """
    decryptor = Cipher(
        algorithms.AES(key),
        modes.ECB(),
        backend=default_backend()
    ).decryptor()
    
    plaintext_padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(plaintext_padded) + unpadder.finalize()
    return plaintext.decode()

password = "your_password_here"
plaintext = "your_message_here"
fixed_salt = b'coeus_lab' 

aes_key = generate_aes_key_from_password(password, fixed_salt)

ciphertext = encrypt_with_aes_ecb(plaintext, aes_key)

decrypted_text = decrypt_with_aes_ecb(ciphertext, aes_key)

print("Original:", plaintext)
print("Encrypted:", ciphertext)
print("Decrypted:", decrypted_text)
