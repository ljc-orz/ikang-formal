ALTER TABLE hospital 
ADD COLUMN economic_zone ENUM('东部', '中部', '西部', '东北') 
DEFAULT NULL 
COMMENT '四大经济板块（东部/中部/西部/东北）';

UPDATE hospital 
SET economic_zone = CASE province
    -- 东部地区（10个省/直辖市）
    WHEN '北京市' THEN '东部'
    WHEN '天津市' THEN '东部'
    WHEN '河北省' THEN '东部'
    WHEN '上海市' THEN '东部'
    WHEN '江苏省' THEN '东部'
    WHEN '浙江省' THEN '东部'
    WHEN '福建省' THEN '东部'
    WHEN '山东省' THEN '东部'
    WHEN '广东省' THEN '东部'
    WHEN '海南省' THEN '东部'   -- 你的数据中没有，但保留以备后续
    
    -- 中部地区（6个省）
    WHEN '山西省' THEN '中部'
    WHEN '安徽省' THEN '中部'
    WHEN '江西省' THEN '中部'
    WHEN '河南省' THEN '中部'
    WHEN '湖北省' THEN '中部'
    WHEN '湖南省' THEN '中部'
    
    -- 西部地区（12个省/区/市）
    WHEN '内蒙古自治区' THEN '西部'
    WHEN '广西壮族自治区' THEN '西部'  -- 保留
    WHEN '重庆市' THEN '西部'
    WHEN '四川省' THEN '西部'
    WHEN '贵州省' THEN '西部'
    WHEN '云南省' THEN '西部'      -- 保留
    WHEN '西藏自治区' THEN '西部'   -- 保留
    WHEN '陕西省' THEN '西部'
    WHEN '甘肃省' THEN '西部'      -- 保留
    WHEN '青海省' THEN '西部'      -- 保留
    WHEN '宁夏回族自治区' THEN '西部'
    WHEN '新疆维吾尔自治区' THEN '西部' -- 保留
    
    -- 东北地区（3个省）
    WHEN '辽宁省' THEN '东北'
    WHEN '吉林省' THEN '东北'
    WHEN '黑龙江省' THEN '东北'    -- 保留
    
    ELSE NULL
END;

-- 以下把检查结果表与医院地区表合并

-- 1. 删除已存在的同名新表
DROP TABLE IF EXISTS result_merged_wide_filtered;

-- 2. 复制原表结构
CREATE TABLE result_merged_wide_filtered LIKE result_merged_wide;

-- 3. 添加 economic_zone 列（类型与 hospital 表一致）
ALTER TABLE result_merged_wide_filtered
    ADD COLUMN economic_zone ENUM('东部','中部','西部','东北') NULL AFTER result_wbc;

-- 4. 插入满足条件的数据，同时关联 hospital 获取 economic_zone
INSERT INTO result_merged_wide_filtered
SELECT r.*, h.economic_zone
FROM result_merged_wide r
JOIN hospital h ON r.hospid = h.hospid
WHERE r.result_alt IS NOT NULL
  AND r.result_bmi IS NOT NULL
  AND r.result_fbg IS NOT NULL
  AND r.result_hb IS NOT NULL
  AND r.result_hba1c IS NOT NULL
  AND r.result_hct IS NOT NULL
  AND r.result_hdl_c IS NOT NULL
  AND r.result_rbc IS NOT NULL
  AND r.result_scr IS NOT NULL
  AND r.result_tg IS NOT NULL
  AND r.result_wbc IS NOT NULL
  AND r.grade_1 IN ('GOOD', 'USABLE')
  AND r.grade_2 IN ('GOOD', 'USABLE');