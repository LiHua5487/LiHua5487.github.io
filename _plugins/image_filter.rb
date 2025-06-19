module Jekyll
    module ObsidianImageFilter
      def obsidian_image(input)
        # 匹配 Obsidian 图片语法：![[path|widthxheight]]
        input.gsub(/!\[\[([^\|\]]+)(?:\|(\d+)x(\d+))?\]\]/) do
          path = $1.gsub(/\s+/, '%20')  # 替换空格为 URL 编码
          width = $2
          height = $3
          
          # 构建图片标签
          if width && height
            "<img src=\"#{path}\" alt=\"#{File.basename(path)}\" width=\"#{width}\" height=\"#{height}\" style=\"max-width:100%;height:auto;\">"
          else
            "<img src=\"#{path}\" alt=\"#{File.basename(path)}\" style=\"max-width:100%;height:auto;\">"
          end
        end
      end
    end
  end
  
  Liquid::Template.register_filter(Jekyll::ObsidianImageFilter)
  