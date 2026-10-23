njs_int_t
njs_vmcode_interpreter(njs_vm_t *vm, u_char *pc, void *promise_cap,
    void *async_ctx)
{
    u_char                       *catch;
    double                       num, exponent;
    int32_t                      i32;
    uint32_t                     u32;
    njs_str_t                    string;
    njs_uint_t                   hint;
    njs_bool_t                   valid, lambda_call;
    njs_value_t                  *retval, *value1, *value2;
    njs_value_t                  *src, *s1, *s2, dst;
    njs_value_t                  *function, name;
    njs_value_t                  numeric1, numeric2, primitive1, primitive2;
    njs_frame_t                  *frame;
    njs_jump_off_t               ret;
    njs_vmcode_await_t           *await;
    njs_native_frame_t           *previous, *native;
    njs_property_next_t          *next;
    njs_vmcode_import_t          *import;
    njs_vmcode_finally_t         *finally;
    njs_vmcode_generic_t         *vmcode;
    njs_vmcode_variable_t        *var;
    njs_vmcode_move_arg_t        *move_arg;
    njs_vmcode_prop_get_t        *get;
    njs_vmcode_prop_set_t        *set;
    njs_vmcode_operation_t       op;
    njs_vmcode_prop_next_t       *pnext;
    njs_vmcode_test_jump_t       *test_jump;
    njs_vmcode_equal_jump_t      *equal;
    njs_vmcode_try_return_t      *try_return;
    njs_vmcode_method_frame_t    *method_frame;
    njs_vmcode_function_copy_t   *fcopy;
    njs_vmcode_prop_accessor_t   *accessor;
    njs_vmcode_try_trampoline_t  *try_trampoline;
    njs_vmcode_function_frame_t  *function_frame;

#ifdef NJS_OPCODE_DEBUG
    njs_vmcode_debug(vm, pc, "ENTER");
#endif

next:

    for ( ;; ) {

        vmcode = (njs_vmcode_generic_t *) pc;

        /*
         * The first operand is passed as is in value2 to
         *   NJS_VMCODE_JUMP,
         *   NJS_VMCODE_IF_TRUE_JUMP,
         *   NJS_VMCODE_IF_FALSE_JUMP,
         *   NJS_VMCODE_FUNCTION_FRAME,
         *   NJS_VMCODE_FUNCTION_CALL,
         *   NJS_VMCODE_RETURN,
         *   NJS_VMCODE_TRY_START,
         *   NJS_VMCODE_TRY_CONTINUE,
         *   NJS_VMCODE_TRY_BREAK,
         *   NJS_VMCODE_TRY_END,
         *   NJS_VMCODE_CATCH,
         *   NJS_VMCODE_THROW,
         *   NJS_VMCODE_STOP.
         */
        value2 = (njs_value_t *) vmcode->operand1;
        value1 = NULL;

        switch (vmcode->code.operands) {

        case NJS_VMCODE_3OPERANDS:
            njs_vmcode_operand(vm, vmcode->operand3, value2);

            /* Fall through. */

        case NJS_VMCODE_2OPERANDS:
            njs_vmcode_operand(vm, vmcode->operand2, value1);
        }

        op = vmcode->code.operation;

        /*
         * On success an operation returns size of the bytecode,
         * a jump offset or zero after the call or return operations.
         * Jumps can return a negative offset.  Compilers can generate
         *    (ret < 0 && ret >= NJS_PREEMPT)
         * as a single unsigned comparision.
         */

#ifdef NJS_OPCODE_DEBUG
        njs_disassemble(pc, NULL, 1, NULL);
#endif

        if (op > NJS_VMCODE_NORET) {

            if (op == NJS_VMCODE_MOVE) {
                njs_vmcode_operand(vm, vmcode->operand1, retval);
                *retval = *value1;

                pc += sizeof(njs_vmcode_move_t);
                goto next;
            }

            if (op == NJS_VMCODE_PROPERTY_GET) {
                get = (njs_vmcode_prop_get_t *) pc;
                njs_vmcode_operand(vm, get->value, retval);

                ret = njs_value_property(vm, value1, value2, retval);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                pc += sizeof(njs_vmcode_prop_get_t);
                goto next;
            }

            switch (op) {
            case NJS_VMCODE_INCREMENT:
            case NJS_VMCODE_POST_INCREMENT:
            case NJS_VMCODE_DECREMENT:
            case NJS_VMCODE_POST_DECREMENT:
                if (njs_slow_path(!njs_is_numeric(value2))) {
                    ret = njs_value_to_numeric(vm, value2, &numeric1);
                    if (njs_slow_path(ret != NJS_OK)) {
                        goto error;
                    }

                    num = njs_number(&numeric1);

                } else {
                    num = njs_number(value2);
                }

                njs_set_number(value1,
                           num + (1 - 2 * ((op - NJS_VMCODE_INCREMENT) >> 1)));

                njs_vmcode_operand(vm, vmcode->operand1, retval);

                if (op & 1) {
                    njs_set_number(retval, num);

                } else {
                    *retval = *value1;
                }

                pc += sizeof(njs_vmcode_3addr_t);
                goto next;

            case NJS_VMCODE_GLOBAL_GET:
                get = (njs_vmcode_prop_get_t *) pc;
                njs_vmcode_operand(vm, get->value, retval);

                ret = njs_value_property(vm, value1, value2, retval);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                pc += sizeof(njs_vmcode_prop_get_t);

                if (ret == NJS_OK) {
                    pc += sizeof(njs_vmcode_error_t);
                }

                goto next;

            /*
             * njs_vmcode_try_return() saves a return value to use it later by
             * njs_vmcode_finally(), and jumps to the nearest try_break block.
             */
            case NJS_VMCODE_TRY_RETURN:
                njs_vmcode_operand(vm, vmcode->operand1, retval);
                *retval = *value1;

                try_return = (njs_vmcode_try_return_t *) pc;
                pc += try_return->offset;
                goto next;

            case NJS_VMCODE_LESS:
            case NJS_VMCODE_GREATER:
            case NJS_VMCODE_LESS_OR_EQUAL:
            case NJS_VMCODE_GREATER_OR_EQUAL:
            case NJS_VMCODE_ADDITION:
                if (njs_slow_path(!njs_is_primitive(value1))) {
                    hint = (op == NJS_VMCODE_ADDITION) && njs_is_date(value1);
                    ret = njs_value_to_primitive(vm, &primitive1, value1, hint);
                    if (ret != NJS_OK) {
                        goto error;
                    }

                    value1 = &primitive1;
                }

                if (njs_slow_path(!njs_is_primitive(value2))) {
                    hint = (op == NJS_VMCODE_ADDITION) && njs_is_date(value2);
                    ret = njs_value_to_primitive(vm, &primitive2, value2, hint);
                    if (ret != NJS_OK) {
                        goto error;
                    }

                    value2 = &primitive2;
                }

                if (njs_slow_path(njs_is_symbol(value1)
                                  || njs_is_symbol(value2)))
                {
                    njs_symbol_conversion_failed(vm,
                        (op == NJS_VMCODE_ADDITION) &&
                        (njs_is_string(value1) || njs_is_string(value2)));

                    goto error;
                }

                njs_vmcode_operand(vm, vmcode->operand1, retval);

                if (op == NJS_VMCODE_ADDITION) {
                    if (njs_fast_path(njs_is_numeric(value1)
                                      && njs_is_numeric(value2)))
                    {
                        njs_set_number(retval, njs_number(value1)
                                               + njs_number(value2));
                        pc += sizeof(njs_vmcode_3addr_t);
                        goto next;
                    }

                    if (njs_is_string(value1)) {
                        s1 = value1;
                        s2 = &dst;
                        src = value2;

                    } else {
                        s1 = &dst;
                        s2 = value2;
                        src = value1;
                    }

                    ret = njs_primitive_value_to_string(vm, &dst, src);
                    if (njs_slow_path(ret != NJS_OK)) {
                        goto error;
                    }

                    ret = njs_string_concat(vm, s1, s2);
                    if (njs_slow_path(ret == NJS_ERROR)) {
                        goto error;
                    }

                    *retval = vm->retval;

                    pc += ret;
                    goto next;
                }

                if ((uint8_t) (op - NJS_VMCODE_GREATER) < 2) {
                    /* NJS_VMCODE_GREATER, NJS_VMCODE_LESS_OR_EQUAL */
                    src = value1;
                    value1 = value2;
                    value2 = src;
                }

                ret = njs_primitive_values_compare(vm, value1, value2);

                if (op < NJS_VMCODE_LESS_OR_EQUAL) {
                    ret = ret > 0;

                } else {
                    ret = ret == 0;
                }

                njs_set_boolean(retval, ret);

                pc += sizeof(njs_vmcode_3addr_t);
                goto next;

            case NJS_VMCODE_EQUAL:
            case NJS_VMCODE_NOT_EQUAL:
                ret = njs_values_equal(vm, value1, value2);
                if (njs_slow_path(ret < 0)) {
                    goto error;
                }

                ret ^= op - NJS_VMCODE_EQUAL;

                njs_vmcode_operand(vm, vmcode->operand1, retval);
                njs_set_boolean(retval, ret);

                pc += sizeof(njs_vmcode_3addr_t);
                goto next;

            case NJS_VMCODE_SUBSTRACTION:
            case NJS_VMCODE_MULTIPLICATION:
            case NJS_VMCODE_EXPONENTIATION:
            case NJS_VMCODE_DIVISION:
            case NJS_VMCODE_REMAINDER:
            case NJS_VMCODE_BITWISE_AND:
            case NJS_VMCODE_BITWISE_OR:
            case NJS_VMCODE_BITWISE_XOR:
            case NJS_VMCODE_LEFT_SHIFT:
            case NJS_VMCODE_RIGHT_SHIFT:
            case NJS_VMCODE_UNSIGNED_RIGHT_SHIFT:
                if (njs_slow_path(!njs_is_numeric(value1))) {
                    ret = njs_value_to_numeric(vm, value1, &numeric1);
                    if (njs_slow_path(ret != NJS_OK)) {
                        goto error;
                    }

                    value1 = &numeric1;
                }

                if (njs_slow_path(!njs_is_numeric(value2))) {
                    ret = njs_value_to_numeric(vm, value2, &numeric2);
                    if (njs_slow_path(ret != NJS_OK)) {
                        goto error;
                    }

                    value2 = &numeric2;
                }

                num = njs_number(value1);

                njs_vmcode_operand(vm, vmcode->operand1, retval);
                pc += sizeof(njs_vmcode_3addr_t);

                switch (op) {
                case NJS_VMCODE_SUBSTRACTION:
                    num -= njs_number(value2);
                    break;

                case NJS_VMCODE_MULTIPLICATION:
                    num *= njs_number(value2);
                    break;

                case NJS_VMCODE_EXPONENTIATION:
                    exponent = njs_number(value2);

                    /*
                     * According to ES7:
                     *  1. If exponent is NaN, the result should be NaN;
                     *  2. The result of +/-1 ** +/-Infinity should be NaN.
                     */
                    valid = njs_expect(1, fabs(num) != 1
                                          || (!isnan(exponent)
                                              && !isinf(exponent)));

                    num = valid ? pow(num, exponent) : NAN;
                    break;

                case NJS_VMCODE_DIVISION:
                    num /= njs_number(value2);
                    break;

                case NJS_VMCODE_REMAINDER:
                    num = fmod(num, njs_number(value2));
                    break;

                case NJS_VMCODE_BITWISE_AND:
                case NJS_VMCODE_BITWISE_OR:
                case NJS_VMCODE_BITWISE_XOR:
                    i32 = njs_number_to_int32(njs_number(value2));

                    switch (op) {
                    case NJS_VMCODE_BITWISE_AND:
                        i32 &= njs_number_to_int32(num);
                        break;

                    case NJS_VMCODE_BITWISE_OR:
                        i32 |= njs_number_to_int32(num);
                        break;

                    case NJS_VMCODE_BITWISE_XOR:
                        i32 ^= njs_number_to_int32(num);
                        break;
                    }

                    njs_set_int32(retval, i32);
                    goto next;

                default:
                    u32 = njs_number_to_uint32(njs_number(value2)) & 0x1f;

                    switch (op) {
                    case NJS_VMCODE_LEFT_SHIFT:
                    case NJS_VMCODE_RIGHT_SHIFT:
                        i32 = njs_number_to_int32(num);

                        if (op == NJS_VMCODE_LEFT_SHIFT) {
                            /* Shifting of negative numbers is undefined. */
                            i32 = (uint32_t) i32 << u32;
                        } else {
                            i32 >>= u32;
                        }

                        njs_set_int32(retval, i32);
                        break;

                    default: /* NJS_VMCODE_UNSIGNED_RIGHT_SHIFT */
                        njs_set_uint32(retval,
                                       njs_number_to_uint32(num) >> u32);
                    }

                    goto next;
                }

                njs_set_number(retval, num);
                goto next;

            case NJS_VMCODE_OBJECT_COPY:
                ret = njs_vmcode_object_copy(vm, value1, value2);
                break;

            case NJS_VMCODE_TEMPLATE_LITERAL:
                ret = njs_vmcode_template_literal(vm, value1, value2);
                break;

            case NJS_VMCODE_PROPERTY_IN:
                ret = njs_vmcode_property_in(vm, value1, value2);
                break;

            case NJS_VMCODE_PROPERTY_DELETE:
                ret = njs_value_property_delete(vm, value1, value2, NULL);
                if (njs_fast_path(ret != NJS_ERROR)) {
                    vm->retval = njs_value_true;

                    ret = sizeof(njs_vmcode_3addr_t);
                }

                break;

            case NJS_VMCODE_PROPERTY_FOREACH:
                ret = njs_vmcode_property_foreach(vm, value1, value2, pc);
                break;

            case NJS_VMCODE_STRICT_EQUAL:
            case NJS_VMCODE_STRICT_NOT_EQUAL:
                ret = njs_values_strict_equal(value1, value2);

                ret ^= op - NJS_VMCODE_STRICT_EQUAL;

                njs_vmcode_operand(vm, vmcode->operand1, retval);
                njs_set_boolean(retval, ret);

                pc += sizeof(njs_vmcode_3addr_t);
                goto next;

            case NJS_VMCODE_TEST_IF_TRUE:
            case NJS_VMCODE_TEST_IF_FALSE:
            case NJS_VMCODE_COALESCE:
                if (op == NJS_VMCODE_COALESCE) {
                    ret = !njs_is_null_or_undefined(value1);

                } else {
                    ret = njs_is_true(value1);
                    ret ^= op - NJS_VMCODE_TEST_IF_TRUE;
                }

                if (ret) {
                    test_jump = (njs_vmcode_test_jump_t *) pc;
                    ret = test_jump->offset;

                } else {
                    ret = sizeof(njs_vmcode_3addr_t);
                }

                njs_vmcode_operand(vm, vmcode->operand1, retval);
                *retval = *value1;

                pc += ret;
                goto next;

            case NJS_VMCODE_UNARY_PLUS:
            case NJS_VMCODE_UNARY_NEGATION:
            case NJS_VMCODE_BITWISE_NOT:
                if (njs_slow_path(!njs_is_numeric(value1))) {
                    ret = njs_value_to_numeric(vm, value1, &numeric1);
                    if (njs_slow_path(ret != NJS_OK)) {
                        goto error;
                    }

                    value1 = &numeric1;
                }

                num = njs_number(value1);
                njs_vmcode_operand(vm, vmcode->operand1, retval);

                switch (op) {
                case NJS_VMCODE_UNARY_NEGATION:
                    num = -num;

                    /* Fall through. */
                case NJS_VMCODE_UNARY_PLUS:
                    njs_set_number(retval, num);
                    break;

                case NJS_VMCODE_BITWISE_NOT:
                    njs_set_int32(retval, ~njs_number_to_uint32(num));
                }

                pc += sizeof(njs_vmcode_2addr_t);
                goto next;

            case NJS_VMCODE_LOGICAL_NOT:
                njs_vmcode_operand(vm, vmcode->operand1, retval);
                njs_set_boolean(retval, !njs_is_true(value1));

                pc += sizeof(njs_vmcode_2addr_t);
                goto next;

            case NJS_VMCODE_OBJECT:
                ret = njs_vmcode_object(vm);
                break;

            case NJS_VMCODE_ARRAY:
                ret = njs_vmcode_array(vm, pc);
                break;

            case NJS_VMCODE_FUNCTION:
                ret = njs_vmcode_function(vm, pc);
                break;

            case NJS_VMCODE_REGEXP:
                ret = njs_vmcode_regexp(vm, pc);
                break;

            case NJS_VMCODE_INSTANCE_OF:
                ret = njs_vmcode_instance_of(vm, value1, value2);
                break;

            case NJS_VMCODE_TYPEOF:
                ret = njs_vmcode_typeof(vm, value1, value2);
                break;

            case NJS_VMCODE_VOID:
                njs_set_undefined(&vm->retval);

                ret = sizeof(njs_vmcode_2addr_t);
                break;

            case NJS_VMCODE_DELETE:
                njs_release(vm, value1);
                vm->retval = njs_value_true;

                ret = sizeof(njs_vmcode_2addr_t);
                break;

            case NJS_VMCODE_DEBUGGER:
                ret = njs_vmcode_debugger(vm);
                break;

            default:
                njs_internal_error(vm, "%d has retval", op);
                goto error;
            }

            if (njs_slow_path(ret < 0 && ret >= NJS_PREEMPT)) {
                break;
            }

            njs_vmcode_operand(vm, vmcode->operand1, retval);
            njs_release(vm, retval);
            *retval = vm->retval;

        } else {

            switch (op) {
            case NJS_VMCODE_MOVE_ARG:
                move_arg = (njs_vmcode_move_arg_t *) pc;
                native = vm->top_frame;

                hint = move_arg->dst;

                value1 = &native->arguments_offset[hint];
                njs_vmcode_operand(vm, move_arg->src, value2);

                *value1 = *value2;

                ret = sizeof(njs_vmcode_move_arg_t);
                break;

            case NJS_VMCODE_STOP:
                njs_vmcode_operand(vm, (njs_index_t) value2, value2);
                vm->retval = *value2;

#ifdef NJS_OPCODE_DEBUG
                njs_vmcode_debug(vm, pc, "EXIT STOP");
#endif

                return NJS_OK;

            case NJS_VMCODE_JUMP:
                ret = (njs_jump_off_t) value2;
                break;

            case NJS_VMCODE_PROPERTY_SET:
                set = (njs_vmcode_prop_set_t *) pc;
                njs_vmcode_operand(vm, set->value, retval);

                ret = njs_value_property_set(vm, value1, value2, retval);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                ret = sizeof(njs_vmcode_prop_set_t);
                break;

            case NJS_VMCODE_PROPERTY_ACCESSOR:
                accessor = (njs_vmcode_prop_accessor_t *) pc;
                njs_vmcode_operand(vm, accessor->value, function);

                ret = njs_value_to_key(vm, &name, value2);
                if (njs_slow_path(ret != NJS_OK)) {
                    njs_internal_error(vm, "failed conversion of type \"%s\" "
                                       "to string while property define",
                                       njs_type_string(value2->type));
                    goto error;
                }

                ret = njs_object_prop_define(vm, value1, &name, function,
                                             accessor->type);
                if (njs_slow_path(ret != NJS_OK)) {
                    return NJS_ERROR;
                }

                ret = sizeof(njs_vmcode_prop_accessor_t);
                break;

            case NJS_VMCODE_IF_TRUE_JUMP:
            case NJS_VMCODE_IF_FALSE_JUMP:
                ret = njs_is_true(value1);

                ret ^= op - NJS_VMCODE_IF_TRUE_JUMP;

                ret = ret ? (njs_jump_off_t) value2
                          : (njs_jump_off_t) sizeof(njs_vmcode_cond_jump_t);

                break;

            case NJS_VMCODE_IF_EQUAL_JUMP:
                if (njs_values_strict_equal(value1, value2)) {
                    equal = (njs_vmcode_equal_jump_t *) pc;
                    ret = equal->offset;

                } else {
                    ret = sizeof(njs_vmcode_3addr_t);
                }

                break;

            case NJS_VMCODE_PROPERTY_INIT:
                set = (njs_vmcode_prop_set_t *) pc;
                njs_vmcode_operand(vm, set->value, retval);
                ret = njs_vmcode_property_init(vm, value1, value2, retval);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                break;

            case NJS_VMCODE_RETURN:
                njs_vmcode_operand(vm, (njs_index_t) value2, value2);

#ifdef NJS_OPCODE_DEBUG
                njs_vmcode_debug(vm, pc, "EXIT RETURN");
#endif

                return njs_vmcode_return(vm, NULL, value2);

            case NJS_VMCODE_FUNCTION_COPY:
                fcopy = (njs_vmcode_function_copy_t *) pc;
                ret = njs_vmcode_function_copy(vm, fcopy->function,
                                               fcopy->retval);
                break;

            case NJS_VMCODE_FUNCTION_FRAME:
                function_frame = (njs_vmcode_function_frame_t *) pc;

                /* TODO: external object instead of void this. */

                ret = njs_function_frame_create(vm, value1,
                                                &njs_value_undefined,
                                                (uintptr_t) value2,
                                                function_frame->ctor);

                if (njs_slow_path(ret != NJS_OK)) {
                    goto error;
                }

                ret = sizeof(njs_vmcode_function_frame_t);
                break;

            case NJS_VMCODE_METHOD_FRAME:
                method_frame = (njs_vmcode_method_frame_t *) pc;

                ret = njs_value_property(vm, value1, value2, &dst);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                if (njs_slow_path(!njs_is_function(&dst))) {
                    ret = njs_value_to_key(vm, value2, value2);
                    if (njs_slow_path(ret != NJS_OK)) {
                        return NJS_ERROR;
                    }

                    njs_key_string_get(vm, value2, &string);
                    njs_type_error(vm,
                               "(intermediate value)[\"%V\"] is not a function",
                               &string);
                    goto error;
                }

                ret = njs_function_frame_create(vm, &dst, value1,
                                                method_frame->nargs,
                                                method_frame->ctor);

                if (njs_slow_path(ret != NJS_OK)) {
                    goto error;
                }

                ret = sizeof(njs_vmcode_method_frame_t);
                break;

            case NJS_VMCODE_FUNCTION_CALL:
                vm->active_frame->native.pc = pc;

                njs_vmcode_operand(vm, (njs_index_t) value2, value2);

                ret = njs_function_frame_invoke(vm, value2);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                ret = sizeof(njs_vmcode_function_call_t);
                break;

            case NJS_VMCODE_PROPERTY_NEXT:
                pnext = (njs_vmcode_prop_next_t *) pc;
                retval = njs_scope_value(vm, pnext->retval);

                next = value2->data.u.next;

                if (next->index < next->array->length) {
                    *retval = next->array->start[next->index++];

                    ret = pnext->offset;
                    break;
                }

                njs_mp_free(vm->mem_pool, next);

                ret = sizeof(njs_vmcode_prop_next_t);
                break;

            case NJS_VMCODE_ARGUMENTS:
                ret = njs_vmcode_arguments(vm, pc);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                break;

            case NJS_VMCODE_PROTO_INIT:
                set = (njs_vmcode_prop_set_t *) pc;
                njs_vmcode_operand(vm, set->value, retval);
                ret = njs_vmcode_proto_init(vm, value1, value2, retval);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                break;

            case NJS_VMCODE_IMPORT:
                import = (njs_vmcode_import_t *) pc;
                retval = njs_scope_value(vm, import->retval);
                ret = njs_vmcode_import(vm, import->module, retval);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                break;

            case NJS_VMCODE_AWAIT:
                await = (njs_vmcode_await_t *) pc;

#ifdef NJS_OPCODE_DEBUG
                njs_vmcode_debug(vm, pc, "EXIT AWAIT");
#endif

                return njs_vmcode_await(vm, await, promise_cap, async_ctx);

            case NJS_VMCODE_TRY_START:
                ret = njs_vmcode_try_start(vm, value1, value2, pc);
                if (njs_slow_path(ret == NJS_ERROR)) {
                    goto error;
                }

                break;

            case NJS_VMCODE_THROW:
                njs_vmcode_operand(vm, (njs_index_t) value2, value2);
                vm->retval = *value2;
                goto error;

            case NJS_VMCODE_TRY_BREAK:
                try_trampoline = (njs_vmcode_try_trampoline_t *) pc;
                value1 = njs_scope_value(vm, try_trampoline->exit_value);

                ret = njs_vmcode_try_break(vm, value1, value2);
                break;

            case NJS_VMCODE_TRY_CONTINUE:
                try_trampoline = (njs_vmcode_try_trampoline_t *) pc;
                value1 = njs_scope_value(vm, try_trampoline->exit_value);

                ret = njs_vmcode_try_continue(vm, value1, value2);
                break;

            case NJS_VMCODE_TRY_END:
                ret = njs_vmcode_try_end(vm, value1, value2);
                break;

            /*
             * njs_vmcode_catch() is set on the start of a "catch" block to
             * store exception and to remove a "try" block if there is no
             * "finally" block or to update a catch address to the start of
             * a "finally" block.
             * njs_vmcode_catch() is set on the start of a "finally" block
             * to store uncaught exception and to remove a "try" block.
             */
            case NJS_VMCODE_CATCH:
                *value1 = vm->retval;

                if ((njs_jump_off_t) value2 == sizeof(njs_vmcode_catch_t)) {
                    ret = njs_vmcode_try_end(vm, value1, value2);

                } else {
                    frame = (njs_frame_t *) vm->top_frame;
                    frame->exception.catch = pc + (njs_jump_off_t) value2;
                    ret = sizeof(njs_vmcode_catch_t);
                }

                break;

            case NJS_VMCODE_FINALLY:
                finally = (njs_vmcode_finally_t *) pc;
                value1 = njs_scope_value(vm, finally->exit_value);

                ret = njs_vmcode_finally(vm, value1, value2, pc);

                switch (ret) {
                case NJS_OK:

#ifdef NJS_OPCODE_DEBUG
                    njs_vmcode_debug(vm, pc, "EXIT FINALLY");
#endif

                    return NJS_OK;
                case NJS_ERROR:
                    goto error;
                }

                break;

            case NJS_VMCODE_LET:
                var = (njs_vmcode_variable_t *) pc;
                value1 = njs_scope_value(vm, var->dst);

                if (njs_is_valid(value1)) {
                    value1 = njs_mp_alloc(vm->mem_pool, sizeof(njs_value_t));
                    if (njs_slow_path(value1 == NULL)) {
                        return NJS_ERROR;
                    }

                    njs_scope_value_set(vm, var->dst, value1);
                }

                njs_set_undefined(value1);

                ret = sizeof(njs_vmcode_variable_t);
                break;

            case NJS_VMCODE_LET_UPDATE:
                var = (njs_vmcode_variable_t *) pc;
                value2 = njs_scope_value(vm, var->dst);

                value1 = njs_mp_alloc(vm->mem_pool, sizeof(njs_value_t));
                if (njs_slow_path(value1 == NULL)) {
                    return NJS_ERROR;
                }

                *value1 = *value2;

                njs_scope_value_set(vm, var->dst, value1);

                ret = sizeof(njs_vmcode_variable_t);
                break;

            case NJS_VMCODE_INITIALIZATION_TEST:
                var = (njs_vmcode_variable_t *) pc;
                value1 = njs_scope_value(vm, var->dst);

                if (njs_is_valid(value1)) {
                    ret = sizeof(njs_vmcode_variable_t);
                    break;
                }

                /* Fall through. */

            case NJS_VMCODE_NOT_INITIALIZED:
                njs_reference_error(vm, "cannot access variable "
                                        "before initialization");
                goto error;

            case NJS_VMCODE_ERROR:
                njs_vmcode_error(vm, pc);
                goto error;

            case NJS_VMCODE_ASSIGNMENT_ERROR:
                njs_type_error(vm, "assignment to constant variable");
                goto error;

            default:
                njs_internal_error(vm, "%d has NO retval", op);
                goto error;
            }
        }

        pc += ret;
    }

error:

    if (njs_is_error(&vm->retval)) {
        vm->active_frame->native.pc = pc;
        (void) njs_error_stack_attach(vm, &vm->retval);
    }

    for ( ;; ) {
        native = vm->top_frame;

        if (!native->native) {
            frame = (njs_frame_t *) native;
            catch = frame->exception.catch;

            if (catch != NULL) {
                pc = catch;

                goto next;
            }
        }

        previous = native->previous;
        if (previous == NULL) {
            break;
        }

        lambda_call = (native == &vm->active_frame->native);

        njs_vm_scopes_restore(vm, native, previous);

        if (native->size != 0) {
            vm->stack_size -= native->size;
            njs_mp_free(vm->mem_pool, native);
        }

        if (lambda_call) {
            break;
        }
    }

#ifdef NJS_OPCODE_DEBUG
    njs_vmcode_debug(vm, pc, "EXIT ERROR");
#endif

    return NJS_ERROR;
}
