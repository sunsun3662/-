
// 定义目标架构
.arch armv8-a
// 代码段
.text
// 数据段
.data

// 定义字符串常量
.section .rodata
.align 3
input_format:
    .string "%d"
output_format:
    .string "%d\n"

// BSS 段（未初始化数据）
.bss
.align 2
a:
    .space 4    // int a
b:
    .space 4    // int b
i:
    .space 4    // int i
t:
    .space 4    // int t
n:
    .space 4    // int n

// 代码段
.text
.align 2

// 主函数
.global main
.type main, %function
main:
    // 函数序言：保存帧指针和返回地址
    stp x29, x30, [sp, -16]!
    mov x29, sp

    // 初始化变量 a = 0
    adrp x0, a
    add x0, x0, :lo12:a
    mov w1, #0
    str w1, [x0]

    // 初始化变量 b = 1
    adrp x0, b
    add x0, x0, :lo12:b
    mov w1, #1
    str w1, [x0]

    // 初始化变量 i = 1
    adrp x0, i
    add x0, x0, :lo12:i
    mov w1, #1
    str w1, [x0]

    // 读取输入 n
    // 调用 printf 提示输入
    adrp x0, input_prompt
    add x0, x0, :lo12:input_prompt
    bl printf

    // 调用 scanf 读取 n
    adrp x0, input_format
    add x0, x0, :lo12:input_format
    adrp x1, n
    add x1, x1, :lo12:n
    bl scanf

    // 输出 a (0)
    adrp x0, a
    add x0, x0, :lo12:a
    ldr w1, [x0]
    adrp x0, output_format
    add x0, x0, :lo12:output_format
    bl printf

    // 输出 b (1)
    adrp x0, b
    add x0, x0, :lo12:b
    ldr w1, [x0]
    adrp x0, output_format
    add x0, x0, :lo12:output_format
    bl printf

    // 进入 while 循环
    b while_condition

while_condition:
    // 比较 i < n
    adrp x0, i
    add x0, x0, :lo12:i
    ldr w1, [x0]           // w1 = i
    adrp x0, n
    add x0, x0, :lo12:n
    ldr w2, [x0]           // w2 = n
    cmp w1, w2
    b.ge while_exit        // 如果 i >= n，退出循环

    // 循环体开始
    // t = b
    adrp x0, b
    add x0, x0, :lo12:b
    ldr w1, [x0]           // w1 = b
    adrp x0, t
    add x0, x0, :lo12:t
    str w1, [x0]           // t = b

    // b = a + b
    adrp x0, a
    add x0, x0, :lo12:a
    ldr w1, [x0]           // w1 = a
    adrp x0, b
    add x0, x0, :lo12:b
    ldr w2, [x0]           // w2 = b
    add w1, w1, w2         // w1 = a + b
    str w1, [x0]           // b = a + b

    // 输出新的 b
    adrp x0, output_format
    add x0, x0, :lo12:output_format
    // w1 中已经是 b 的值
    bl printf

    // a = t
    adrp x0, t
    add x0, x0, :lo12:t
    ldr w1, [x0]           // w1 = t
    adrp x0, a
    add x0, x0, :lo12:a
    str w1, [x0]           // a = t

    // i = i + 1
    adrp x0, i
    add x0, x0, :lo12:i
    ldr w1, [x0]           // w1 = i
    add w1, w1, #1         // i = i + 1
    str w1, [x0]           // 存回 i

    // 继续循环
    b while_condition

while_exit:
    // 设置返回值为 0
    mov w0, #0

    // 函数尾声：恢复帧指针和返回地址
    ldp x29, x30, [sp], 16
    ret

// 输入提示字符串
.section .rodata
.align 3
input_prompt:
    .string "请输入 n: "

// 声明外部函数
.global printf
.global scanf
