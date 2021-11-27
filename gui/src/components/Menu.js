import React from 'react';
import { Typography, Menu as VarnishMenu, Icon } from '@allenai/varnish';
import {Layout } from 'antd'
import { InternalLink } from '../components/InternalLink'
import { modelGroups } from '../models'

const { BodySmall } = Typography;
const { IconMenuItemColumns, Item, SubMenu } = VarnishMenu;
const { Sider } = Layout;
const { ImgIcon } = Icon;

/*******************************************************************************
  <Menu /> Component
*******************************************************************************/

export default class Menu extends React.Component {
  siderWidthExpanded = '300px';
  siderWidthCollapsed = '80px';
  constructor(props) {
      super(props);

      this.state = {
          menuCollapsed: false
      };
  }

  handleMenuCollapse = () => {
      this.setState({ menuCollapsed: !this.state.menuCollapsed });
  };

  render() {
    return (
      <Sider
        width={this.siderWidthExpanded}
        collapsedWidth={this.siderWidthCollapsed}
        collapsible
        collapsed={this.state.menuCollapsed}
        onCollapse={this.handleMenuCollapse}
        >
          <VarnishMenu
            defaultSelectedKeys={[this.props.redirectedModel]}
            defaultOpenKeys={modelGroups.filter(g => g.defaultOpen).map(g => g.label)}
            mode="inline"
          >
            {modelGroups.map(g => (
              <SubMenu
                key={g.label}
                title={
                  <IconMenuItemColumns>
                    {g.iconSrc && (
                      <ImgIcon src={g.iconSrc} />
                    )}
                    <BodySmall>{g.label}</BodySmall>
                  </IconMenuItemColumns>
                }
              >
                {g.models.map(m => (
                  <Item key={m.model}>
                    <InternalLink to={"/" + m.model} onClick={() => {}}>
                      <span>{m.name}</span>
                    </InternalLink>
                  </Item>
                ))}
              </SubMenu>
            ))}
          </VarnishMenu>
      </Sider>
    )
  }
}
