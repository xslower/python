--select concat_ws('|', collect())
-- 去除字段两边的标点符号
regexp_replace(content,"^([^\\w\\u4E00-\\u9FFF]|_)+|([^\\w\\u4E00-\\u9FFF]|_)+$","")
-- 获取当前日期
select SUBSTR(FROM_UNIXTIME(UNIX_TIMESTAMP()-3600*24),1,10)

SELECT
  A.exp_group_id,
  A.content,
  A.show_pv,
  A.click_pv,
  A.ctr,
  G.exp_group_id,
  cast(((G.ctr - A.ctr)/A.ctr*100) as decimal(10, 2)) AS diff_G,
  G.content,
  G.show_pv,
  G.click_pv,
  G.ctr,
  H.exp_group_id,
  cast(((H.ctr - A.ctr)/A.ctr*100) as decimal(10, 2)) AS diff_H,
  H.content,
  H.show_pv,
  H.click_pv,
  H.ctr,
  I.exp_group_id,
  cast(((I.ctr - A.ctr)/A.ctr*100) as decimal(10, 2)) AS diff_I,
  I.content,
  I.show_pv,
  I.click_pv,
  I.ctr,
  J.exp_group_id,
  cast(((J.ctr - A.ctr)/A.ctr*100) as decimal(10, 2)) AS diff_J,
  J.content,
  J.show_pv,
  J.click_pv,
  J.ctr
FROM
  (
    SELECT
      exp_group_id,
      content,
      show_pv,
      click_pv,
      cast((click_pv / show_pv*100) as decimal(10, 2)) AS ctr
    From
      qjpdm.abt_content_emoticon_di
    WHERE
      exp_group_id = '31858'
      and thedate = SUBSTR(FROM_UNIXTIME(UNIX_TIMESTAMP()-3600*24),1,10)
    order by
      show_pv desc
    limit
      1000
  ) AS A,
  (
    SELECT
      exp_group_id,
      content,
      show_pv,
      click_pv,
      cast((click_pv / show_pv*100) as decimal(10, 2)) AS ctr
    From
      qjpdm.abt_content_emoticon_di
    WHERE
      exp_group_id = '31864'
      and thedate = SUBSTR(FROM_UNIXTIME(UNIX_TIMESTAMP()-3600*24),1,10)
    order by
      show_pv desc
    limit
      1000
  ) AS G,
  (
    SELECT
      exp_group_id,
      content,
      show_pv,
      click_pv,
      cast((click_pv / show_pv*100) as decimal(10, 2)) AS ctr
    From
      qjpdm.abt_content_emoticon_di
    WHERE
      exp_group_id = '31865'
      and thedate = SUBSTR(FROM_UNIXTIME(UNIX_TIMESTAMP()-3600*24),1,10)
    order by
      show_pv desc
    limit
      1000
  ) AS H,
  (
    SELECT
      exp_group_id,
      content,
      show_pv,
      click_pv,
      cast((click_pv / show_pv*100) as decimal(10, 2)) AS ctr
    From
      qjpdm.abt_content_emoticon_di
    WHERE
      exp_group_id = '31866'
      and thedate = SUBSTR(FROM_UNIXTIME(UNIX_TIMESTAMP()-3600*24),1,10)
    order by
      show_pv desc
    limit
      1000
  ) AS I,
  (
    SELECT
      exp_group_id,
      content,
      show_pv,
      click_pv,
      cast((click_pv / show_pv*100) as decimal(10, 2)) AS ctr
    From
      qjpdm.abt_content_emoticon_di
    WHERE
      exp_group_id = '31867'
      and thedate = SUBSTR(FROM_UNIXTIME(UNIX_TIMESTAMP()-3600*24),1,10)
    order by
      show_pv desc
    limit
      1000
  ) AS J
WHERE
   A.content = G.content
  and A.content = H.content
  and A.content = I.content
  and A.content = J.content
order by
  A.show_pv desc
  limit 1000