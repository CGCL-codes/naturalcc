int flb_config_map_set(struct mk_list *properties, struct mk_list *map, void *context)
{
    int ret;
    int numentries;
    char *base;
    char *m_bool;
    int *m_i_num;
    double *m_d_num;
    size_t *m_s_num;
    flb_sds_t *m_str;
    struct flb_kv *kv;
    struct mk_list *head;
    struct mk_list *m_head;
    struct mk_list **m_list;
    struct mk_list *list;
    struct flb_config_map *m = NULL;
    struct flb_config_map_val *entry = NULL;

    base = context;

    /* Link 'already processed default values' into the caller context */
    mk_list_foreach(m_head, map) {
        m = mk_list_entry(m_head, struct flb_config_map, _head);

        /*
         * If the map type allows multiple entries, the user context is a pointer
         * for a linked list. We just point their structure to our pre-processed
         * list of entries.
         */
        if (m->flags & FLB_CONFIG_MAP_MULT && m->set_property == FLB_TRUE) {
            m_list = (struct mk_list **) (base + m->offset);
            *m_list = m->value.mult;
            continue;
        }

        /*
         * If no default value exists or the map will not write to the user
         * context.. skip it.
         */
        if (!m->def_value || m->set_property == FLB_FALSE) {
            continue;
        }

        /*
         * If a property set by the user will override the default value, just
         * do not put the default value into the context since it will be replaced
         * later.
         */
        ret = properties_override_default(properties, m->name);
        if (ret == FLB_TRUE) {
            continue;
        }

        /* All the following steps are direct writes to the user context */
        if (m->type == FLB_CONFIG_MAP_STR) {
            m_str = (char **) (base + m->offset);
            *m_str = m->value.val.str;
        }
        else if (m->type == FLB_CONFIG_MAP_INT) {
            m_i_num = (int *) (base + m->offset);
            *m_i_num = m->value.val.i_num;
        }
        else if (m->type == FLB_CONFIG_MAP_DOUBLE) {
            m_d_num = (double *) (base + m->offset);
            *m_d_num = m->value.val.d_num;
        }
        else if (m->type == FLB_CONFIG_MAP_SIZE) {
            m_s_num = (size_t *) (base + m->offset);
            *m_s_num = m->value.val.s_num;
        }
        else if (m->type == FLB_CONFIG_MAP_TIME) {
            m_i_num = (int *) (base + m->offset);
            *m_i_num = m->value.val.s_num;
        }
        else if (m->type == FLB_CONFIG_MAP_BOOL) {
            m_bool = (char *) (base + m->offset);
            *m_bool = m->value.val.boolean;
        }
        else if (m->type >= FLB_CONFIG_MAP_CLIST ||
                 m->type <= FLB_CONFIG_MAP_SLIST_4) {
            m_list = (struct mk_list **) (base + m->offset);
            *m_list = m->value.val.list;
        }
    }

    /*
     * Iterate all properties coming from the configuration reader. If a property overrides
     * a default value already set in the previous step, just link to the new value.
     */
    mk_list_foreach(head, properties) {
        kv = mk_list_entry(head, struct flb_kv, _head);
        if (kv->val == NULL) {
            continue;
        }

        mk_list_foreach(m_head, map) {
            m = mk_list_entry(m_head, struct flb_config_map, _head);
            if (flb_sds_len(kv->key) != flb_sds_len(m->name)) {
                m = NULL;
                continue;
            }

            if (strncasecmp(kv->key, m->name, flb_sds_len(m->name)) == 0) {
                break;
            }
            m = NULL;
            continue;

        }

        if (!m || m->set_property == FLB_FALSE) {
            continue;
        }

        /* Check if the map allows multiple entries */
        if (m->flags & FLB_CONFIG_MAP_MULT) {
            /* Create node */
            entry = flb_calloc(1, sizeof(struct flb_config_map_val));
            if (!entry) {
                flb_errno();
                return -1;
            }

            /* Populate value */
            if 
            // Assign the value from the configuration key-value pair to the appropriate field
            // in the entry structure based on the map type. Handle different data types
            // such as string, integer, double, size, time, boolean, and list. Convert the
            // string value to the required type using utility functions if necessary.
            // For boolean types, validate the value and handle errors by logging and
            // freeing resources. For list types, parse the string into a list, validate
            // the number of entries, and handle errors similarly. Ensure the entry is
            // added to the map's list of multiple values if the map allows multiple entries.
            // <MASK>

            /* Add entry to the map 'mult' list tail */
            mk_list_add(&entry->_head, m->value.mult);

            /* Override user context */
            m_list = (struct mk_list **) (base + m->offset);
            *m_list = m->value.mult;
        }
        else if (map != NULL) {
            /* Direct write to user context */
            if (m->type == FLB_CONFIG_MAP_STR) {
                m_str = (char **) (base + m->offset);
                *m_str = kv->val;
            }
            else if (m->type == FLB_CONFIG_MAP_INT) {
                m_i_num = (int *) (base + m->offset);
                *m_i_num = atoi(kv->val);
            }
            else if (m->type == FLB_CONFIG_MAP_DOUBLE) {
                m_d_num = (double *) (base + m->offset);
                *m_d_num = atof(kv->val);
            }
            else if (m->type == FLB_CONFIG_MAP_BOOL) {
                m_bool = (char *) (base + m->offset);
                ret = flb_utils_bool(kv->val);
                if (ret == -1) {
                    flb_error("[config map] invalid value for boolean property '%s=%s'",
                              m->name, kv->val);
                    return -1;
                }
                *m_bool = ret;
            }
            else if (m->type == FLB_CONFIG_MAP_SIZE) {
                m_s_num = (size_t *) (base + m->offset);
                *m_s_num = flb_utils_size_to_bytes(kv->val);
            }
            else if (m->type == FLB_CONFIG_MAP_TIME) {
                m_i_num = (int *) (base + m->offset);
                *m_i_num = flb_utils_time_to_seconds(kv->val);
            }
            else if (m->type >= FLB_CONFIG_MAP_CLIST ||
                     m->type <= FLB_CONFIG_MAP_SLIST_4) {
                list = parse_string_map_to_list(m, kv->val);
                if (!list) {
                    flb_error("[config map] cannot parse list of values '%s'", kv->val);
                    flb_free(entry);
                    return -1;
                }

                if (m->value.val.list) {
                    destroy_map_val(m->type, &m->value);
                }

                m->value.val.list = list;
                m_list = (struct mk_list **) (base + m->offset);
                *m_list = m->value.val.list;
            }
        }
    }

    return 0;
}
