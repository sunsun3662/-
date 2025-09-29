
; === 函数声明 ===
declare i32 @getint()          ; 读取整数
declare void @putint(i32)      ; 输出整数
declare void @putch(i32)       ; 输出字符（用于换行）

; === 全局常量定义（用于换行符）===
@newline = constant [2 x i8] c"\0A\00"    ; 换行符

; === 主函数定义 ===
define i32 @main() {
entry:
    ; 分配局部变量：a, b, i, t, n
    %a = alloca i32, align 4
    %b = alloca i32, align 4
    %i = alloca i32, align 4
    %t = alloca i32, align 4
    %n = alloca i32, align 4
    
    ; 初始化变量
    store i32 0, i32* %a, align 4      ; a = 0
    store i32 1, i32* %b, align 4      ; b = 1
    store i32 1, i32* %i, align 4      ; i = 1
    
    ; 读取 n
    %n_val = call i32 @getint()
    store i32 %n_val, i32* %n, align 4
    
    ; 输出 a (0)
    %a_val1 = load i32, i32* %a, align 4
    call void @putint(i32 %a_val1)
    call void @putch(i32 10)  ; 换行
    
    ; 输出 b (1)
    %b_val1 = load i32, i32* %b, align 4
    call void @putint(i32 %b_val1)
    call void @putch(i32 10)  ; 换行
    
    ; 进入 while 循环判断
    br label %while_cond

while_cond:
    ; 比较 i < n
    %i_val = load i32, i32* %i, align 4
    %n_val_cond = load i32, i32* %n, align 4
    %cmp = icmp slt i32 %i_val, %n_val_cond  ; i < n
    br i1 %cmp, label %while_body, label %exit

while_body:
    ; t = b
    %b_val_body = load i32, i32* %b, align 4
    store i32 %b_val_body, i32* %t, align 4
    
    ; b = a + b
    %a_val_body = load i32, i32* %a, align 4
    %b_val_temp = load i32, i32* %b, align 4
    %new_b = add i32 %a_val_body, %b_val_temp
    store i32 %new_b, i32* %b, align 4
    
    ; 输出新的 b
    call void @putint(i32 %new_b)
    call void @putch(i32 10)  ; 换行
    
    ; a = t
    %t_val = load i32, i32* %t, align 4
    store i32 %t_val, i32* %a, align 4
    
    ; i = i + 1
    %i_val_inc = load i32, i32* %i, align 4
    %new_i = add i32 %i_val_inc, 1
    store i32 %new_i, i32* %i, align 4
    
    ; 跳回循环判断
    br label %while_cond

exit:
    ret i32 0
}
