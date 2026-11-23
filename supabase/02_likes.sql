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
