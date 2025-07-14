from jittor.utils.pytorch_converter import convert

pytorch_code="""


使用代码替换这行

"""

jittor_code = convert(pytorch_code)
print(jittor_code)