-- Nomes colaborativos da galeria Detach! 26.
-- Rodar inteiro no SQL Editor do Supabase (pode rodar de novo: é idempotente).
--
-- Modelo de acesso: as tabelas ficam fechadas (RLS sem nenhuma política), e o site
-- só fala com o banco por funções. Assim quem tem a chave pública consegue sugerir,
-- votar e pedir remoção, mas não consegue ler, apagar ou editar nada direto.
--
-- Regras:
--   * um nome aparece para todos assim que é sugerido (saldo de votos >= 1), ou quando
--     você marca a sugestão como 'approved' no painel (Table Editor → name_suggestions);
--   * com mais de um nome, vale o de maior saldo; saldo <= 0 tira do ar e <= -2 some
--     da lista de votação; 'hidden' esconde de vez;
--   * "remover meu nome" bloqueia a pessoa na hora (nenhum nome aparece e novas
--     sugestões são recusadas). Para desfazer, apague a linha em blocked_people.

create table if not exists public.name_suggestions (
  id          bigint generated always as identity primary key,
  person      int  not null check (person between 0 and 999),
  name        text not null,
  name_key    text not null,               -- versão normalizada, para juntar iguais
  status      text not null default 'pending' check (status in ('pending', 'approved', 'hidden')),
  client_id   uuid not null,
  created_at  timestamptz not null default now(),
  unique (person, name_key)
);

create table if not exists public.name_votes (
  suggestion_id bigint not null references public.name_suggestions(id) on delete cascade,
  client_id     uuid   not null,
  value         smallint not null check (value in (-1, 1)),
  created_at    timestamptz not null default now(),
  primary key (suggestion_id, client_id)
);

create table if not exists public.blocked_people (
  person      int primary key,
  reason      text,
  client_id   uuid,
  created_at  timestamptz not null default now()
);

alter table public.name_suggestions enable row level security;
alter table public.name_votes       enable row level security;
alter table public.blocked_people   enable row level security;

-- placar de cada sugestão
create or replace view public.suggestion_scores with (security_invoker = true) as
select s.*, coalesce(sum(v.value), 0)::int as score, count(v.*)::int as votes
from public.name_suggestions s
left join public.name_votes v on v.suggestion_id = s.id
group by s.id;

-- normaliza o nome digitado (espaços, tamanho, links)
create or replace function public._clean_name(p text) returns text
language sql immutable as $$
  select nullif(regexp_replace(btrim(coalesce(p, '')), '\s+', ' ', 'g'), '')
$$;

-- nomes que aparecem para todos: o melhor de cada pessoa
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

-- sugestões de uma pessoa, para votar (inclui as ainda não confirmadas)
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

-- sugerir um nome (se já existe igual para a pessoa, vira um voto a favor)
create or replace function public.suggest_name(p_person int, p_name text, p_client uuid)
returns text
language plpgsql security definer set search_path = public as $$
declare
  v_name text := _clean_name(p_name);
  v_key  text;
  v_id   bigint;
begin
  if p_client is null then return 'erro'; end if;
  if v_name is null or char_length(v_name) < 2 or char_length(v_name) > 40 then return 'tamanho'; end if;
  if v_name ~* '(https?://|www\.|\.com|@)' then return 'invalido'; end if;
  if exists (select 1 from blocked_people where person = p_person) then return 'bloqueado'; end if;
  -- limite simples contra abuso: 40 sugestões por navegador por hora
  if (select count(*) from name_suggestions where client_id = p_client and created_at > now() - interval '1 hour') >= 40 then
    return 'limite';
  end if;
  v_key := lower(translate(v_name, 'áàâãäéèêëíìîïóòôõöúùûüçÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ', 'aaaaaeeeeiiiiooooouuuucaaaaaeeeeiiiiooooouuuuc'));
  insert into name_suggestions (person, name, name_key, client_id)
  values (p_person, v_name, v_key, p_client)
  on conflict (person, name_key) do nothing
  returning id into v_id;
  if v_id is null then
    select id into v_id from name_suggestions where person = p_person and name_key = v_key;
  end if;
  insert into name_votes (suggestion_id, client_id, value) values (v_id, p_client, 1)
  on conflict (suggestion_id, client_id) do update set value = 1, created_at = now();
  return 'ok';
end $$;

-- votar (+1 / -1; 0 retira o voto)
create or replace function public.vote_name(p_suggestion bigint, p_client uuid, p_value int)
returns text
language plpgsql security definer set search_path = public as $$
begin
  if p_client is null or p_value not in (-1, 0, 1) then return 'erro'; end if;
  if p_value = 0 then
    delete from name_votes where suggestion_id = p_suggestion and client_id = p_client;
  else
    insert into name_votes (suggestion_id, client_id, value) values (p_suggestion, p_client, p_value)
    on conflict (suggestion_id, client_id) do update set value = excluded.value, created_at = now();
  end if;
  return 'ok';
end $$;

-- "remover meu nome": esconde na hora
create or replace function public.request_removal(p_person int, p_reason text, p_client uuid)
returns text
language plpgsql security definer set search_path = public as $$
begin
  insert into blocked_people (person, reason, client_id)
  values (p_person, left(coalesce(p_reason, ''), 300), p_client)
  on conflict (person) do nothing;
  return 'ok';
end $$;

-- só as funções ficam abertas ao público
revoke all on public.name_suggestions, public.name_votes, public.blocked_people, public.suggestion_scores from anon, authenticated;
revoke all on function public._clean_name(text) from public, anon, authenticated;
grant execute on function public.get_names() to anon, authenticated;
grant execute on function public.get_suggestions(int, uuid) to anon, authenticated;
grant execute on function public.suggest_name(int, text, uuid) to anon, authenticated;
grant execute on function public.vote_name(bigint, uuid, int) to anon, authenticated;
grant execute on function public.request_removal(int, text, uuid) to anon, authenticated;

-- ---------------------------------------------------------------------------
-- Curtidas nas fotos (um coração por navegador por foto)

create table if not exists public.photo_likes (
  photo_id    text not null check (photo_id ~ '^[A-Za-z0-9_-]{10,80}$'),
  client_id   uuid not null,
  created_at  timestamptz not null default now(),
  primary key (photo_id, client_id)
);
alter table public.photo_likes enable row level security;
create index if not exists photo_likes_photo on public.photo_likes (photo_id);

-- contagem de todas as fotos curtidas (só as que têm ao menos um coração)
create or replace function public.get_like_counts()
returns table (photo_id text, likes int)
language sql stable security definer set search_path = public as $$
  select photo_id, count(*)::int from photo_likes group by photo_id
$$;

-- fotos que este navegador curtiu
create or replace function public.get_my_likes(p_client uuid)
returns table (photo_id text)
language sql stable security definer set search_path = public as $$
  select photo_id from photo_likes where client_id = p_client
$$;

-- curtir (p_on = true) ou descurtir; devolve o total atualizado da foto
create or replace function public.like_photo(p_photo text, p_client uuid, p_on boolean)
returns int
language plpgsql security definer set search_path = public as $$
begin
  if p_client is null or p_photo !~ '^[A-Za-z0-9_-]{10,80}$' then return -1; end if;
  if p_on then
    -- limite simples contra abuso: 300 corações por navegador por hora
    if (select count(*) from photo_likes where client_id = p_client and created_at > now() - interval '1 hour') >= 300 then
      return -2;
    end if;
    insert into photo_likes (photo_id, client_id) values (p_photo, p_client) on conflict do nothing;
  else
    delete from photo_likes where photo_id = p_photo and client_id = p_client;
  end if;
  return (select count(*)::int from photo_likes where photo_id = p_photo);
end $$;

revoke all on public.photo_likes from anon, authenticated;
grant execute on function public.get_like_counts() to anon, authenticated;
grant execute on function public.get_my_likes(uuid) to anon, authenticated;
grant execute on function public.like_photo(text, uuid, boolean) to anon, authenticated;
