-- View: public.secteurs_par_article

-- DROP VIEW public.secteurs_par_article;

CREATE OR REPLACE VIEW public.secteurs_par_article
 AS
 SELECT rss.id,
    sec.name,
    rss.titre,
    rss.lien,
    rss.date
   FROM articles_rss rss
     LEFT JOIN article_sectors asec ON rss.id = asec.article_id
     LEFT JOIN sectors sec ON asec.sector_id = sec.id
  ORDER BY rss.id, asec.sector_id;

ALTER TABLE public.secteurs_par_article
    OWNER TO manu;

