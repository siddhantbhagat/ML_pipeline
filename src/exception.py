import sys
from src.logger import logging

def error_message_details(error,error_detail:sys):
    _,_,error_tb = error_detail.exc_info()
    file_name = error_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in file {file_name} at line no {error_tb.tb_lineno} error: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)

    def __str__(self):
        return self.error_message


# to test exception.py
# if __name__ == '__main__':
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info('Zero division error')
#         raise CustomException(e,sys)