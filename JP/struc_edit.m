function funcs = struc_edit
  funcs.search_row = @search_row;
  funcs.insert = @insert;
end
function idx = search_row(struc, fields, vals)
    if isempty(fieldnames(struc))
        idx = [];
    else
        bool_vec = ones(length(struc),1,'logical');
        for kk = 1:length(vals)
            val = vals{kk};
            field = fields{kk};

            if isnumeric(val)
                bool_vec = bool_vec & [struc.(field)]';
            else
                bool_vec = bool_vec & strcmp({struc.(field)}',val);
            end

        end
        idx = find(bool_vec);
    end
end
function struc = insert(struc, row, idx)
    if isempty(fieldnames(struc))
        first = true;
        for ff = fieldnames(row)
            if first
                struc(end+1).(ff) = row.(ff);
                first= false;
            else
                struc(end).(ff) = row.(ff);
            end
        end
    else
        fldname = fieldnames(row);
        for ff = 1:length(fldname)
            fld = fldname{ff};
            struc(idx).(fld) = row.(fld);
        end
    end
end