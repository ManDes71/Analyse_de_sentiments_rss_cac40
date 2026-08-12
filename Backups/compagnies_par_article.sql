CREATE OR REPLACE VIEW public.compagnies_par_article
 AS
 SELECT rss.id,
    com.name,
    rss.titre,
    rss.lien,
    rss.date
   FROM articles_rss rss
     LEFT JOIN article_companies ac ON rss.id = ac.article_id
     LEFT JOIN companies com ON ac.company_id = com.id
  ORDER BY rss.id, ac.article_id;

ALTER TABLE public.compagnies_par_article
    OWNER TO manu;
