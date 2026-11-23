-- Nome aparece assim que é sugerido (saldo >= 1). Rodar no SQL Editor.

create or replace function public.get_names()
returns table (person int, name text, score int)
language sql stable security definer set search_path = public as $$
  select distinct on (s.person) s.person, s.name, s.score
  from suggestion_scores s
  where s.status <> 'hidden'
    and (s.status = 'approved' or s.score >= 1)
    and not exists (select 1 from blocked_people b where b.person = s.person)
  order by s.person, (s.status = 'approved') desc, s.score desc, s.created_at
$$;

create or replace function public.get_suggestions(p_person int, p_client uuid)
returns table (id bigint, name text, score int, my_vote smallint, visible boolean, blocked boolean)
language sql stable security definer set search_path = public as $$
  select s.id, s.name, s.score,
         (select v.value from name_votes v where v.suggestion_id = s.id and v.client_id = p_client),
         (s.status = 'approved' or s.score >= 1),
         false
  from suggestion_scores s
  where s.person = p_person and s.status <> 'hidden' and s.score > -2
    and not exists (select 1 from blocked_people b where b.person = p_person)
  union all
  select null, null, null, null, false, true
  where exists (select 1 from blocked_people b where b.person = p_person)
  order by 3 desc nulls last
  limit 8
$$;
