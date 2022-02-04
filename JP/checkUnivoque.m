function [bool] = checkUnivoque(path_to_list)
bool = true;
brk = false;
load(path_to_list)
brain_area_group_list = cell(length(neuron_id),1);
animal_name_list = cell(length(neuron_id),1);
date_list = cell(length(neuron_id),1);
session_list = cell(length(neuron_id),1);
for k =1:length(neuron_id)
    splt = split(paths_to_fit{k},'\');
    path_k =  splt{end};
    path_k = path_k(1:(regexp(path_k,'\.')-1));
    splt = split(path_k,'_');
    brain_area_group_list{k} = splt{1};
    animal_name_list{k} = splt{2};
    date_list{k} = splt{3};
    session_list{k} = splt{4};
end

for animal = unique(animal_name_list)'
    name = animal{1};
    sel = strcmp(animal_name_list,name);
    brain_area_anim = brain_area_group_list(sel);
    date_list_anim = date_list(sel);
    session_list_anim = session_list(sel);
    neuron_id_anim = neuron_id(sel);
    use_coupling_anim = use_coupling(sel);
    for ba = unique(brain_area_anim)
        name = ba{1};
        sel = strcmp(brain_area_anim,name);
        if sum(sel)==0
            continue
        end
        date_list_ba = date_list_anim(sel);
        session_list_ba = session_list_anim(sel);
        neuron_id_ba = neuron_id_anim(sel);
        use_coupling_ba = use_coupling_anim(sel);
        for date = unique(date_list_ba)
            
            name = date{1};
            sel = strcmp(date_list_ba,name);
            if sum(sel)==0
                continue
            end
            session_list_date = session_list_ba(sel);
            neuron_id_date = neuron_id_ba(sel);
            use_coupling_date = use_coupling_ba(sel);
            for sess = unique(session_list_date)
                name = sess{1};
                sel = strcmp(session_list_date,name);
                if sum(sel)==0
                    continue
                end
                neuron_id_sess = neuron_id_date(sel);
                use_coupling_sess = use_coupling_date(sel);
                for neu = unique(neuron_id_sess)
                    
                    sel = neuron_id_sess == neu;
                    if sum(sel)==0
                        continue
                    end
                    use_coupling_neu = use_coupling_sess(sel);
                    for bl = unique(use_coupling_neu)
                        bool = sum(use_coupling_neu == bl) == 1;
                        if ~bool
                            brk = true;
                        end
                        if brk
                            break
                        end
                    end
                    if brk
                        break
                    end
                end
                if brk
                    break
                end
            end
            if brk
                break
            end
        end
        if brk
            break
        end
    end
    if brk
        break
    end
end
end