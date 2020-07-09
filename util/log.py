import logging

logging.basicConfig(level=logging.DEBUG, filename="out.log", filemode="a",
                    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s")

if __name__ == '__main__':
    logging.info("hello...")
